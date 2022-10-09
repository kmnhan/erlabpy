"""Plotting utilities."""
import pkgutil
from io import StringIO

import cmasher
import cmocean
import colorcet
import matplotlib
import matplotlib.colors
import matplotlib.font_manager
import numpy as np

__all__ = [
    "annotations",
    "bz",
    "colors",
    "general",
    "interactive",
]


def refresh_fonts(silent=True):
    for path in matplotlib.font_manager.findSystemFonts():
        try:
            matplotlib.font_manager.fontManager.addfont(path)
        except OSError as exc:
            if not silent:
                print("Failed to open font file %s: %s", path, exc)
        except Exception as exc:
            if not silent:
                print("Failed to extract font properties from %s: %s", path, exc)


def load_igor_ct(file, name):
    file = pkgutil.get_data(__package__, "IgorCT/" + file)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name, np.genfromtxt(StringIO(file.decode())) / 65535
    )
    matplotlib.colormaps.register(cmap)
    matplotlib.colormaps.register(cmap.reversed())


refresh_fonts()
load_igor_ct("Blue-White.txt", "Igor_BlueWhite")
