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
   utils

"""

__all__ = ["correct_with_edge", "quick_resolution", "shift", "slice_along_path"]

from erlab.analysis import fit, gold, image, interpolate, mask, transform  # noqa: F401
from erlab.analysis.gold import correct_with_edge, quick_resolution
from erlab.analysis.interpolate import slice_along_path
from erlab.analysis.utils import shift
