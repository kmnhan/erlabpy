"""Convenient access to various plotting functions."""

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
    "property_label",
    "proportional_colorbar",
    "scale_units",
    "set_titles",
    "set_xlabels",
    "set_ylabels",
    "sizebar",
    "unify_clim",
]

import numpy as np

from erlab.plotting.annotations import (
    copy_mathtext,
    fancy_labels,
    label_subplot_properties,
    label_subplots,
    label_subplots_nature,
    mark_points,
    mark_points_outside,
    plot_hv_text,
    property_label,
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


def integer_ticks(axes):
    if np.iterable(axes):
        for ax in np.asarray(axes, dtype=object):
            integer_ticks(ax)
        return
    axes.set_xticks(
        [
            t
            for t in axes.get_xticks()
            if t.is_integer() and t >= axes.get_xlim()[0] and t <= axes.get_xlim()[1]
        ]
    )
    axes.set_yticks(
        [
            t
            for t in axes.get_yticks()
            if t.is_integer() and t >= axes.get_ylim()[0] and t <= axes.get_ylim()[1]
        ]
    )
