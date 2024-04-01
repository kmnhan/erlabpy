"""Convenient access to various plotting functions."""

__all__ = [
    "copy_mathtext",
    "fancy_labels",
    "label_subplot_properties",
    "label_subplots",
    "label_subplots_nature",
    "mark_points",
    "mark_points_outside",
    "plot_hv_text",
    "property_label",
    "scale_units",
    "set_titles",
    "set_xlabels",
    "set_ylabels",
    "sizebar",
    "get_bz_edge",
    "plot_hex_bz",
    "InversePowerNorm",
    "TwoSlopePowerNorm",
    "CenteredPowerNorm",
    "TwoSlopeInversePowerNorm",
    "CenteredInversePowerNorm",
    "get_mappable",
    "unify_clim",
    "flatten_transparency",
    "image_is_light",
    "nice_colorbar",
    "proportional_colorbar",
    "fermiline",
    "figwh",
    "autoscale_to",
    "place_inset",
    "plot_array",
    "plot_array_2d",
    "gradient_fill",
    "plot_slices",
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
    fermiline,
    figwh,
    gradient_fill,
    place_inset,
    plot_array,
    plot_array_2d,
    plot_slices,
)


def clean_labels(axes, tick_right=False, *args, **kwargs):
    if axes.ndim == 1:
        axes = axes[None]
    for ax in axes[:-1, :].flat:
        ax.set_xlabel("")
    if tick_right:
        target = axes[:, :-1]
    else:
        target = axes[:, 1:]
    for ax in target.flat:
        ax.set_ylabel("")
    fancy_labels(axes, *args, **kwargs)
    for ax in axes.flat:
        if tick_right:
            ax.yaxis.set_label_position("right")


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
