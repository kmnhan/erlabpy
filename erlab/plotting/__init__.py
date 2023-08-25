"""
Everything related to plotting.

.. currentmodule:: erlab.plotting

Modules
=======

.. autosummary::
   :toctree: generated
   
   annotations
   bz
   colors
   erplot
   general
   plot3d

"""

__all__ = [
    "annotations",
    "bz",
    "colors",
    "general",
    "plot3d",
]

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


def load_igor_ct(fname, name):
    file = pkgutil.get_data(__package__, "IgorCT/" + fname)
    if fname.endswith(".txt"):
        values = np.genfromtxt(io.StringIO(file.decode()))
    elif fname.endswith(".ibw"):
        values = erlab.io.load_wave(io.BytesIO(file)).values

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name, values / 65535)
    matplotlib.colormaps.register(cmap)
    matplotlib.colormaps.register(cmap.reversed())


load_igor_ct("CTBlueWhite.ibw", "BuWh")
load_igor_ct("CTRainbowLIght.ibw", "RainbowLight")
# load_igor_ct("CTRedTemperature.ibw", "RedTemperature")
load_igor_ct("ColdWarm.ibw", "ColdWarm")
load_igor_ct("BlueHot.ibw", "BlueHot")
# load_igor_ct("PlanetEarth.ibw", "PlanetEarth")
# load_igor_ct("ametrine.ibw", "ametrine")
# load_igor_ct("isolum.ibw", "isolum")
# load_igor_ct("morgenstemning.ibw", "morgenstemning")


matplotlib.style.core.USER_LIBRARY_PATHS.append(
    os.path.join(os.path.dirname(__file__), "stylelib")
)
matplotlib.style.core.reload_library()
