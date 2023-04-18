"""Convenient access to various plotting functions.

"""
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg

from erlab.plotting.annotations import (
    annotate_cuts_erlab,
    copy_mathtext,
    fancy_labels,
    property_label,
    label_subplot_properties,
    label_subplots,
    label_subplots_nature,
    set_titles,
    set_xlabels,
    set_ylabels,
    mark_points_outside,
    mark_points,
    plot_hv_text,
    sizebar,
)
from erlab.plotting.bz import get_bz_edge, plot_hex_bz
from erlab.plotting.colors import (
    InversePowerNorm,
    TwoSlopePowerNorm,
    CenteredPowerNorm,
    TwoSlopeInversePowerNorm,
    CenteredInversePowerNorm,
    get_mappable,
    image_is_light,
    nice_colorbar,
    proportional_colorbar,
)
from erlab.plotting.general import (
    fermiline,
    figwh,
    place_inset,
    plot_array,
    gradient_fill,
    plot_slices,
)
from erlab.interactive.goldtool import goldtool
from erlab.interactive.imagetool import itool
from erlab.interactive.ktool import ktool
from erlab.interactive.noisetool import noisetool

__all__ = [
    "annotate_cuts_erlab",
    "copy_mathtext",
    "fancy_labels",
    "property_label",
    "label_subplot_properties",
    "label_subplots",
    "label_subplots_nature",
    "set_titles",
    "set_xlabels",
    "set_ylabels",
    "mark_points_outside",
    "mark_points",
    "plot_hv_text",
    "sizebar",
    "get_bz_edge",
    "plot_hex_bz",
    "InversePowerNorm",
    "TwoSlopePowerNorm",
    "CenteredPowerNorm",
    "TwoSlopeInversePowerNorm",
    "CenteredInversePowerNorm",
    "get_mappable",
    "image_is_light",
    "nice_colorbar",
    "proportional_colorbar",
    "fermiline",
    "figwh",
    "place_inset",
    "plot_array",
    "gradient_fill",
    "plot_slices",
    "goldtool",
    "itool",
    "ktool",
    "noisetool",
]


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


def autoscale_to(arr, margin=0.2):
    mn, mx = min(arr), max(arr)
    diff = margin * (mx - mn)
    return mn - diff, mx + diff
