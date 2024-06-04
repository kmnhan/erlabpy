"""
Everything related to plotting.

.. currentmodule:: erlab.plotting

Modules
=======

.. autosummary::
   :toctree: generated

   annotations
   atoms
   bz
   colors
   erplot
   general
   plot3d

"""

import importlib
import io
import os
import pkgutil

import matplotlib
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.style
import numpy as np

import erlab.io

# Import colormaps if available
if importlib.util.find_spec("cmasher"):
    importlib.import_module("cmasher")
if importlib.util.find_spec("cmocean"):
    importlib.import_module("cmocean")
if importlib.util.find_spec("colorcet"):
    importlib.import_module("colorcet")


def load_igor_ct(fname: str, name: str) -> None:
    """Load a Igor CT wave file (`.ibw`) and register as a matplotlib colormap.

    Parameters
    ----------
    fname
        Path to the Igor CT wave file.
    name
        The name to register the colormap as.

    """
    file = pkgutil.get_data(__package__, "IgorCT/" + fname)

    if file is None:
        raise FileNotFoundError(f"Could not find file {fname}")

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
