import arpes.xarray_extensions
import lmfit
import csaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from erlab.plotting.colors import proportional_colorbar
from erlab.plotting.general import plot_array

__all__ = ["correct_with_edge"]


def correct_with_edge(
    fmap: xr.DataArray,
    modelresult,
    plot=False,
    zero_nans=False,
    shift_coords=True,
    **improps
):

    if isinstance(fmap, xr.Dataset):
        fmap = fmap.spectrum
    if isinstance(modelresult, lmfit.model.ModelResult):
        edge_quad = modelresult.eval(x=fmap.phi)
    elif callable(modelresult):
        edge_quad = modelresult(fmap.phi.values)
    elif isinstance(modelresult, np.ndarray):
        if len(fmap.phi) != len(modelresult):
            raise ValueError("incompatible modelresult dimensions")
        else:
            edge_quad = modelresult
    else:
        raise ValueError(
            "modelresult must be one of "
            "lmfit.model.ModelResult, "
            "and np.ndarray or a callable"
        )

    corrected = fmap.G.shift_by(
        edge_quad, "eV", zero_nans=zero_nans, shift_coords=shift_coords
    )

    if plot is True:
        _, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))

        improps.setdefault("cmap", "copper")

        if fmap.ndim > 2:
            avg_dims = list(fmap.dims)[:]
            avg_dims.remove("phi")
            avg_dims.remove("eV")
            plot_array(fmap.mean(avg_dims), ax=axes[0], **improps)
            plot_array(corrected.mean(avg_dims), ax=axes[1], **improps)
            # fmap.mean(avg_dims).S.plot(ax=axes[0], **improps)
            # corrected.mean(avg_dims).S.plot(ax=axes[1], **improps)
        else:
            plot_array(fmap, ax=axes[0], **improps)
            plot_array(corrected, ax=axes[1], **improps)
            # fmap.S.plot(ax=axes[0], **improps)
            # corrected.S.plot(ax=axes[1], **improps)
        edge_quad.plot(ax=axes[0], ls="--", color="0.35")

        proportional_colorbar(ax=axes[0])
        proportional_colorbar(ax=axes[1])
        axes[0].set_title("Data")
        axes[1].set_title("Edge Corrected")
    return corrected
