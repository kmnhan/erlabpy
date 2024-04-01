__all__ = ["correct_with_edge"]

import arpes.xarray_extensions # noqa: F401
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from erlab.analysis.fit.models import FermiEdge2dModel
from erlab.plotting.colors import proportional_colorbar
from erlab.plotting.general import plot_array



def correct_with_edge(
    darr: xr.DataArray,
    modelresult,
    plot=False,
    zero_nans=False,
    shift_coords=True,
    **improps,
):
    if isinstance(modelresult, lmfit.model.ModelResult):
        if isinstance(modelresult.model, FermiEdge2dModel):
            edge_quad = xr.DataArray(
                np.polynomial.polynomial.polyval(
                    darr.alpha,
                    np.array(
                        [
                            modelresult.best_values[f"c{i}"]
                            for i in range(modelresult.model.func.poly.degree + 1)
                        ]
                    ),
                ),
                coords=dict(alpha=darr.alpha),
            )
        else:
            edge_quad = modelresult.eval(x=darr.alpha)
            edge_quad = xr.DataArray(
                edge_quad, coords=dict(x=darr.alpha), dims=["alpha"]
            )  # workaround for lmfit 1.22 coercing
    elif callable(modelresult):
        edge_quad = xr.DataArray(
            modelresult(darr.alpha.values), coords=dict(alpha=darr.alpha)
        )
    elif isinstance(modelresult, np.ndarray | xr.DataArray):
        if len(darr.alpha) != len(modelresult):
            raise ValueError("incompatible modelresult dimensions")
        else:
            edge_quad = modelresult
    else:
        raise ValueError(
            "modelresult must be one of "
            "lmfit.model.ModelResult, "
            "and np.ndarray or a callable"
        )

    corrected = darr.G.shift_by(
        edge_quad, "eV", zero_nans=zero_nans, shift_coords=shift_coords
    )

    if plot is True:
        _, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))

        improps.setdefault("cmap", "copper")

        if darr.ndim > 2:
            avg_dims = list(darr.dims)[:]
            avg_dims.remove("alpha")
            avg_dims.remove("eV")
            plot_array(darr.mean(avg_dims), ax=axes[0], **improps)
            plot_array(corrected.mean(avg_dims), ax=axes[1], **improps)
        else:
            plot_array(darr, ax=axes[0], **improps)
            plot_array(corrected, ax=axes[1], **improps)
        edge_quad.plot(ax=axes[0], ls="--", color="0.35")

        proportional_colorbar(ax=axes[0])
        proportional_colorbar(ax=axes[1])
        axes[0].set_title("Data")
        axes[1].set_title("Edge Corrected")
    return corrected
