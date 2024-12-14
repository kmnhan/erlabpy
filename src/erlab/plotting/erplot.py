"""Deprecated convenience module for plotting. Use `erlab.plotting` instead."""

__all__ = [
    "CenteredInversePowerNorm",
    "CenteredPowerNorm",
    "InversePowerNorm",
    "TwoSlopeInversePowerNorm",
    "TwoSlopePowerNorm",
    "autoscale_to",
    "clean_labels",
    "copy_mathtext",
    "fancy_labels",
    "fermiline",
    "figwh",
    "flatten_transparency",
    "get_bz_edge",
    "get_mappable",
    "gradient_fill",
    "image_is_light",
    "integer_ticks",
    "label_subplot_properties",
    "label_subplots",
    "label_subplots_nature",
    "mark_points",
    "mark_points_outside",
    "nice_colorbar",
    "place_inset",
    "plot_array",
    "plot_array_2d",
    "plot_hex_bz",
    "plot_hv_text",
    "plot_slices",
    "property_labels",
    "proportional_colorbar",
    "scale_units",
    "set_titles",
    "set_xlabels",
    "set_ylabels",
    "sizebar",
    "unify_clim",
]


import warnings

from erlab.plotting.annotations import (
    copy_mathtext,
    fancy_labels,
    integer_ticks,
    label_subplot_properties,
    label_subplots,
    label_subplots_nature,
    mark_points,
    mark_points_outside,
    plot_hv_text,
    property_labels,
    scale_units,
    set_titles,
    set_xlabels,
    set_ylabels,
    sizebar,
)
from erlab.plotting.bz import get_bz_edge, plot_hex_bz
from erlab.plotting.colors import (
    CenteredInversePowerNorm,
    CenteredPowerNorm,
    InversePowerNorm,
    TwoSlopeInversePowerNorm,
    TwoSlopePowerNorm,
    flatten_transparency,
    get_mappable,
    image_is_light,
    nice_colorbar,
    proportional_colorbar,
    unify_clim,
)
from erlab.plotting.general import (
    autoscale_to,
    clean_labels,
    fermiline,
    figwh,
    gradient_fill,
    place_inset,
    plot_array,
    plot_array_2d,
    plot_slices,
)

warnings.warn(
    "The convenience `erlab.plotting.erplot` is deprecated and will be removed in the "
    "next major release. Please import as `erlab.plotting` instead.",
    FutureWarning,
    stacklevel=2,
)
