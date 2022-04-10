import arpes.xarray_extensions
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..plotting import plot_array, proportional_colorbar

__all__ = ['correct_with_edge']


def correct_with_edge(fmap: xr.DataArray, modelresult: lmfit.model.ModelResult,
                      plot=False, **improps):

    if isinstance(fmap, xr.Dataset):
        fmap = fmap.spectrum
    edge_quad = modelresult.eval(x=fmap.phi)
    corrected = fmap.G.shift_by(edge_quad - np.amax(edge_quad), "eV")

    if plot is True:
        modelresult.plot()
        _, axes = plt.subplots(1, 2, layout='constrained', figsize=(10, 5))

        improps.setdefault('cmap', 'twilight')

        if fmap.ndim > 2:
            avg_dims = list(fmap.dims)[:]
            avg_dims.remove('phi')
            avg_dims.remove('eV')
            plot_array(fmap.mean(avg_dims), ax=axes[0], **improps)
            plot_array(corrected.mean(avg_dims), ax=axes[1], **improps)
            # fmap.mean(avg_dims).S.plot(ax=axes[0], **improps)
            # corrected.mean(avg_dims).S.plot(ax=axes[1], **improps)
        else:
            plot_array(fmap, ax=axes[0], **improps)
            plot_array(corrected, ax=axes[1], **improps)
            # fmap.S.plot(ax=axes[0], **improps)
            # corrected.S.plot(ax=axes[1], **improps)
        edge_quad.plot(ax=axes[0], ls='--', color='0.35')

        proportional_colorbar(ax=axes[0])
        proportional_colorbar(ax=axes[1])
        axes[0].set_title('Data')
        axes[1].set_title('Edge Corrected')
    return corrected
