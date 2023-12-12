"""Utilities for curve fitting.

.. currentmodule:: erlab.analysis.fit

Modules
=======

.. autosummary::
   :toctree:
   
   functions
   models

"""

__all__ = [
    "ExtendedAffineBroadenedFD",
    "PolynomialModel",
    "MultiPeakModel",
]
from erlab.analysis.fit import functions, models
from erlab.analysis.fit.models import (
    ExtendedAffineBroadenedFD,
    MultiPeakModel,
    PolynomialModel,
)
