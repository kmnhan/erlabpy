"""
Some `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ for convenient
data analysis and visualization.

.. currentmodule:: erlab.accessors

Modules
=======

.. autosummary::
   :toctree: generated

   utils
   kspace
   fit

"""

__all__ = [
    "ModelFitDataArrayAccessor",
    "ModelFitDatasetAccessor",
    "ParallelFitDataArrayAccessor",
    "MomentumAccessor",
    "OffsetView",
    "ImageToolAccessor",
    "PlotAccessor",
    "SelectionAccessor",
]

from erlab.accessors.fit import (
    ModelFitDataArrayAccessor,
    ModelFitDatasetAccessor,
    ParallelFitDataArrayAccessor,
)
from erlab.accessors.kspace import MomentumAccessor, OffsetView
from erlab.accessors.utils import ImageToolAccessor, PlotAccessor, SelectionAccessor
