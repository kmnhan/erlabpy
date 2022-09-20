"""Various functions for data analysis."""
from .correlation import acf2, acf2stack, match_dims, xcorr1d
from .gold import (gold_edge, gold_poly, gold_poly_from_edge, gold_resolution,
                   gold_resolution_roi)
# from .mask import mask_with_hex_bz, mask_with_polygon, polygon_mask
from .transform import *
from .utilities import correct_with_edge
