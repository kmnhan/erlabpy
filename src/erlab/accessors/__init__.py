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

"""  # noqa: D205

__all__ = [
    "InteractiveDataArrayAccessor",
    "InteractiveDatasetAccessor",
    "ModelFitDataArrayAccessor",
    "ModelFitDatasetAccessor",
    "MomentumAccessor",
    "OffsetView",
    "ParallelFitDataArrayAccessor",
    "PlotAccessor",
    "SelectionAccessor",
]

from erlab.accessors.fit import (
    ModelFitDataArrayAccessor,
    ModelFitDatasetAccessor,
    ParallelFitDataArrayAccessor,
)
from erlab.accessors.kspace import MomentumAccessor, OffsetView
from erlab.accessors.utils import (
    InteractiveDataArrayAccessor,
    InteractiveDatasetAccessor,
    PlotAccessor,
    SelectionAccessor,
)
