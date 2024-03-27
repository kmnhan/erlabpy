"""Utilities for curve fitting.

.. currentmodule:: erlab.analysis.fit

Modules
=======

.. autosummary::
   :toctree:

   functions
   models
   spline
   minuit

"""

__all__ = [
    "ExtendedAffineBroadenedFD",
    "PolynomialModel",
    "MultiPeakModel",
    "LeastSq",
    "Minuit",
]

from erlab.analysis.fit.minuit import LeastSq, Minuit
from erlab.analysis.fit.models import (
    ExtendedAffineBroadenedFD,
    MultiPeakModel,
    PolynomialModel,
)
