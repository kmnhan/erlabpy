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
   transform
   interpolate
   utilities
   kspace

"""

__all__ = [
    "gold",
    "correlation",
    "transform",
    "mask_with_polygon",
    "polygon_mask",
    "polygon_mask_points",
    "mask_with_hex_bz",
    "rotateinplane",
    "rotatestackinplane",
    "correct_with_edge",
]

import warnings

from erlab.analysis import fit
from erlab.analysis import gold
from erlab.analysis import correlation
from erlab.analysis import transform
from erlab.analysis.mask import (
    mask_with_hex_bz,
    mask_with_polygon,
    polygon_mask,
    polygon_mask_points,
)
from erlab.analysis.transform import rotateinplane, rotatestackinplane
from erlab.analysis.utilities import correct_with_edge


def gold_edge(*args, **kwargs):
    """:meta private:"""
    import erlab.analysis.gold
    warnings.warn("Use `gold.edge` instead", DeprecationWarning, stacklevel=2)
    return erlab.analysis.gold.edge(*args, **kwargs)


def gold_poly(*args, **kwargs):
    """:meta private:"""
    import erlab.analysis.gold
    warnings.warn("Use `gold.poly` instead", DeprecationWarning, stacklevel=2)
    return erlab.analysis.gold.poly(*args, **kwargs)


def gold_poly_from_edge(*args, **kwargs):
    """:meta private:"""
    import erlab.analysis.gold
    warnings.warn("Use `gold.poly_from_edge` instead", DeprecationWarning, stacklevel=2)
    return erlab.analysis.gold.poly_from_edge(*args, **kwargs)


def gold_resolution(*args, **kwargs):
    """:meta private:"""
    import erlab.analysis.gold
    warnings.warn("Use `gold.resolution` instead", DeprecationWarning, stacklevel=2)
    return erlab.analysis.gold.resolution(*args, **kwargs)


def gold_resolution_roi(*args, **kwargs):
    """:meta private:"""
    import erlab.analysis.gold
    warnings.warn("Use `gold.resolution_roi` instead", DeprecationWarning, stacklevel=2)
    return erlab.analysis.gold.resolution_roi(*args, **kwargs)
