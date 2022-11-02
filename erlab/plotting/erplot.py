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


def clean_labels(axes, *args, **kwargs):
    for ax in axes[:-1,:].flat:
        ax.set_xlabel("")
    for ax in axes[:,1:].flat:
        ax.set_ylabel("")    
    fancy_labels(axes, *args, **kwargs)
    
def autoscale_to(arr, margin=0.2):
    mn, mx = min(arr), max(arr)
    diff = margin * (mx - mn)
    return mn - diff, mx + diff
    