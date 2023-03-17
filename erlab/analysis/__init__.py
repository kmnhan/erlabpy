"""Various functions for data analysis.

"""
from erlab.analysis.correlation import acf2, acf2stack, match_dims, xcorr1d
from erlab.analysis.gold import (
    gold_edge,
    gold_poly,
    gold_poly_from_edge,
    gold_resolution,
    gold_resolution_roi,
)
from erlab.analysis.mask import mask_with_polygon, polygon_mask, mask_with_hex_bz
from erlab.analysis.transform import rotateinplane, rotatestackinplane
from erlab.analysis.utilities import correct_with_edge

__all__ = [
    "acf2",
    "acf2stack",
    "match_dims",
    "xcorr1d",
    "gold_edge",
    "gold_poly",
    "gold_poly_from_edge",
    "gold_resolution",
    "gold_resolution_roi",
    "mask_with_polygon",
    "polygon_mask",
    "mask_with_hex_bz",
    "rotateinplane",
    "rotatestackinplane",
    "correct_with_edge",
]
