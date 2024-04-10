"""
Various functions for data analysis.

.. currentmodule:: erlab.analysis

Modules
=======

.. autosummary::
   :toctree: generated

   fit
   mask
   correlation
   gold
   image
   interpolate
   kspace
   transform
   utilities

"""

__all__ = [
    "correct_with_edge",
    "mask_with_hex_bz",
    "mask_with_polygon",
    "polygon_mask",
    "polygon_mask_points",
    "rotateinplane",
    "rotatestackinplane",
]

from erlab.analysis import correlation, fit, gold, interpolate, kspace  # noqa: F401
from erlab.analysis.mask import (
    mask_with_hex_bz,
    mask_with_polygon,
    polygon_mask,
    polygon_mask_points,
)
from erlab.analysis.transform import rotateinplane, rotatestackinplane
from erlab.analysis.utilities import correct_with_edge
