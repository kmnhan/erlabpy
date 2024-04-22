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
    "shift",
]

from erlab.analysis.gold import correct_with_edge
from erlab.analysis.mask import (
    mask_with_hex_bz,
    mask_with_polygon,
    polygon_mask,
    polygon_mask_points,
)
from erlab.analysis.transform import rotateinplane, rotatestackinplane
from erlab.analysis.utilities import shift
