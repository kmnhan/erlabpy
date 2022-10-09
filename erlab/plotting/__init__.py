"""Plotting utilities."""

import cmasher
import cmocean
import colorcet

import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg

from .annotations import (
    annotate_cuts_erlab,
    copy_mathtext,
    fancy_labels,
    get_si_str,
    label_subplot_properties,
    label_subplots,
    mark_points,
    mark_points_y,
    plot_hv_text,
    refresh_fonts,
    sizebar,
)
from .bz import plot_hex_bz
from .colors import TwoSlopePowerNorm, proportional_colorbar, nice_colorbar, image_is_light
from .general import fermiline, figwh, place_inset, plot_array, plot_slices
from .interactive.goldtool import goldtool
from .interactive.imagetool import itool
from .interactive.imagetool_new import itool_
from .interactive.ktool import ktool
from .interactive.noisetool import noisetool

refresh_fonts()
