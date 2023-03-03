import matplotlib.backends.backend_pdf
import matplotlib.backends.backend_svg

from erlab.plotting.annotations import (
    annotate_cuts_erlab,
    copy_mathtext,
    fancy_labels,
    get_si_str,
    label_subplot_properties,
    label_subplots,
    set_titles,
    mark_points,
    mark_points_y,
    plot_hv_text,
    sizebar,
)
from erlab.plotting.bz import get_bz_edge, plot_hex_bz
from erlab.plotting.colors import (
    InversePowerNorm,
    TwoSlopePowerNorm,
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
    plot_slices,
)
from erlab.plotting.interactive.goldtool import goldtool
from erlab.plotting.interactive.imagetool import itool
from erlab.plotting.interactive.imagetool_new import itool_
from erlab.plotting.interactive.ktool import ktool
from erlab.plotting.interactive.noisetool import noisetool

__all__ = [
    "annotate_cuts_erlab",
    "copy_mathtext",
    "fancy_labels",
    "get_si_str",
    "label_subplot_properties",
    "label_subplots",
    "set_titles",
    "mark_points",
    "mark_points_y",
    "plot_hv_text",
    "sizebar",
    "plot_hex_bz",
    "InversePowerNorm",
    "TwoSlopePowerNorm",
    "get_mappable",
    "image_is_light",
    "nice_colorbar",
    "proportional_colorbar",
    "fermiline",
    "figwh",
    "place_inset",
    "plot_array",
    "plot_slices",
    "goldtool",
    "itool",
    "itool_",
    "ktool",
    "noisetool",
]


def clean_labels(axes, *args, **kwargs):
    if axes.ndim == 1:
        axes = axes[None]
    for ax in axes[:-1, :].flat:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
    fancy_labels(axes, *args, **kwargs)


def autoscale_to(arr, margin=0.2):
    mn, mx = min(arr), max(arr)
    diff = margin * (mx - mn)
    return mn - diff, mx + diff
