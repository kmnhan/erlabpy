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

import io
import os
import pkgutil

import matplotlib
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.style
import numpy as np


def load_igor_ct(
    file: str | os.PathLike | io.BytesIO, name: str, register_reversed: bool = True
) -> None:
    """Load a Igor CT wave file (``.ibw``) and register as a matplotlib colormap.

    Parameters
    ----------
    file
        Path to the color table wave. The wave must have three columns with the red,
        green, and blue values in the range 0-65535.
    name
        The name to register the colormap as.
    register_reversed
        Whether to also register the reversed colormap with the name `name + "_r"`.

    """
    import igor2.binarywave

    values = igor2.binarywave.load(file)["wave"]["wData"]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name, values.astype(np.float64) / 65535
    )
    matplotlib.colormaps.register(cmap)
    if register_reversed:
        matplotlib.colormaps.register(cmap.reversed())


def _get_ct_wave_bytes(file: str) -> io.BytesIO:
    file = pkgutil.get_data(__package__, "IgorCT/" + file)

    if file is None:
        raise FileNotFoundError(f"Could not find file {file}")

    return io.BytesIO(file)


load_igor_ct(_get_ct_wave_bytes("CTBlueWhite.ibw"), "BuWh")
load_igor_ct(_get_ct_wave_bytes("CTRainbowLIght.ibw"), "RainbowLight")
# load_igor_ct(_get_ct_wave_bytes("CTRedTemperature.ibw"), "RedTemperature")
load_igor_ct(_get_ct_wave_bytes("ColdWarm.ibw"), "ColdWarm")
load_igor_ct(_get_ct_wave_bytes("BlueHot.ibw"), "BlueHot")
load_igor_ct(_get_ct_wave_bytes("PlanetEarth.ibw"), "PlanetEarth")
# load_igor_ct(_get_ct_wave_bytes("ametrine.ibw"), "ametrine")
# load_igor_ct(_get_ct_wave_bytes("isolum.ibw"), "isolum")
# load_igor_ct(_get_ct_wave_bytes("morgenstemning.ibw"), "morgenstemning")


matplotlib.style.core.USER_LIBRARY_PATHS.append(
    os.path.join(os.path.dirname(__file__), "stylelib")
)
matplotlib.style.core.reload_library()
