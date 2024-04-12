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
    "FermiEdgeModel",
    "LeastSq",
    "Minuit",
    "MultiPeakModel",
    "PolynomialModel",
]

from erlab.analysis.fit.minuit import LeastSq, Minuit
from erlab.analysis.fit.models import (
    FermiEdgeModel,
    MultiPeakModel,
    PolynomialModel,
)
