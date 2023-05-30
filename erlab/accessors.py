"""
=========================================
xarray accessors (:mod:`erlab.accessors`)
=========================================

.. currentmodule:: erlab.accessors

"""
import erlab.interactive
from erlab.interactive.imagetool import itool
import xarray as xr


class ERLabAccessor:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj


@xr.register_dataset_accessor("er")
class ERLabDataArrayAccessor(ERLabAccessor):
    def show(self, *args, **kwargs):
        return itool(self._obj, *args, **kwargs)


if __name__ == "__main__":
    import numpy as np

    xr.DataArray(np.array([0, 1, 2]))
