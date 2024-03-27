__all__ = ["correct_with_edge"]

import arpes.xarray_extensions # noqa: F401
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from erlab.plotting.colors import proportional_colorbar
from erlab.plotting.general import plot_array
from erlab.analysis.fit.models import FermiEdge2dModel


def correct_with_edge(
    fmap: xr.DataArray,
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
                    fmap.alpha,
                    np.array(
                        [
                            modelresult.best_values[f"c{i}"]
                            for i in range(modelresult.model.func.poly.degree + 1)
                        ]
                    ),
                ),
                coords=dict(alpha=fmap.alpha),
            )
        else:
            edge_quad = modelresult.eval(x=fmap.alpha)
            edge_quad = xr.DataArray(
                edge_quad, coords=dict(x=fmap.alpha), dims=["alpha"]
            )  # workaround for lmfit 1.22 coercing
    elif callable(modelresult):
        edge_quad = xr.DataArray(
            modelresult(fmap.alpha.values), coords=dict(alpha=fmap.alpha)
        )
    elif isinstance(modelresult, np.ndarray | xr.DataArray):
        if len(fmap.alpha) != len(modelresult):
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
            avg_dims.remove("alpha")
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
