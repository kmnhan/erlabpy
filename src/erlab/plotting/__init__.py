"""
Everything related to plotting.

.. currentmodule:: erlab.plotting

The :doc:`Python workflow tutorial </tutorials/python/index>` introduces basic plotting.
For figure preparation tasks, see the :doc:`Plotting How-to guides
</how-to/plotting/index>`.

.. rubric:: Bundled Matplotlib styles

Importing :mod:`erlab.plotting` registers these public style sheets with Matplotlib:

``erlab.general``
    General-purpose defaults for nice figures.
``erlab.nature``
    For Springer Nature journals.
``erlab.physrev``
    For Physical Review journals.
``erlab.arial``
    Arial text and MathText. Requires Arial to be installed.
``erlab.helvetica``
    Helvetica text and MathText. Requires Helvetica to be installed.
``erlab.times``
    Times New Roman text with STIX MathText.
``erlab.stixsans-fallback``
    STIX Sans fallback glyphs for MathText, useful when using ``erlab.helvetica`` or
    ``erlab.arial`` that may not have all glyphs available in the font.

You may combine multiple styles together like ``plt.style.use(["erlab.general",
"erlab.nature", "erlab.arial"])``.

.. rubric:: Modules

This module is organized into several submodules, each providing a different set of
tools for plotting. However, commonly used functions are available directly in the
``erlab.plotting`` namespace, so users should not need to import the submodules
directly.

.. autosummary::
   :toctree: generated

   annotations
   atoms
   bz
   colors
   erplot
   fitting
   general
   plot3d

"""

import io
import os
import pkgutil

import lazy_loader as _lazy
import matplotlib
import matplotlib.colors
import matplotlib.style
import numpy as np

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)


def _load_igor_ct(
    file: str | os.PathLike | io.BytesIO, name: str, register_reversed: bool = True
) -> None:
    """Load an Igor CT wave file (``.ibw``) and register as a matplotlib colormap.

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
    _register_colormap(cmap)
    if register_reversed:
        _register_colormap(cmap.reversed())


def _register_colormap(cmap: matplotlib.colors.Colormap) -> None:
    if cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(cmap)


def _get_ct_wave_bytes(file: str) -> io.BytesIO:
    file = pkgutil.get_data(__package__, "IgorCT/" + file)

    if file is None:
        raise FileNotFoundError(f"Could not find file {file}")

    return io.BytesIO(file)


def _register_style_library() -> None:
    stylelib_path = os.path.join(os.path.dirname(__file__), "stylelib")
    user_library_paths: list[str] | None = getattr(
        matplotlib.style, "USER_LIBRARY_PATHS", None
    )
    if user_library_paths is None:  # pragma: no branch
        # Matplotlib < 3.11 exposes USER_LIBRARY_PATHS only through style.core.
        from matplotlib.style import core as style_core  # pragma: no cover

        user_library_paths = style_core.USER_LIBRARY_PATHS  # pragma: no cover
    if stylelib_path not in user_library_paths:
        user_library_paths.append(stylelib_path)
    matplotlib.style.reload_library()


_load_igor_ct(_get_ct_wave_bytes("CTBlueWhite.ibw"), "BuWh")
_load_igor_ct(_get_ct_wave_bytes("CTRainbowLIght.ibw"), "RainbowLight")
# _load_igor_ct(_get_ct_wave_bytes("CTRedTemperature.ibw"), "RedTemperature")
_load_igor_ct(_get_ct_wave_bytes("ColdWarm.ibw"), "ColdWarm")
_load_igor_ct(_get_ct_wave_bytes("BlueHot.ibw"), "BlueHot")
_load_igor_ct(_get_ct_wave_bytes("PlanetEarth.ibw"), "PlanetEarth")
# _load_igor_ct(_get_ct_wave_bytes("ametrine.ibw"), "ametrine")
# _load_igor_ct(_get_ct_wave_bytes("isolum.ibw"), "isolum")
# _load_igor_ct(_get_ct_wave_bytes("morgenstemning.ibw"), "morgenstemning")
_register_style_library()
