"""Various functions used in fitting.

.. currentmodule:: erlab.analysis.fit.functions

Modules
=======

.. autosummary::
   :toctree:

   dynamic
   general

"""

__all__ = [
    "TINY",
    "do_convolve",
    "do_convolve_y",
    "gaussian_wh",
    "lorentzian_wh",
    "fermi_dirac",
    "fermi_dirac_linbkg",
    "fermi_dirac_linbkg_broad",
    "step_linbkg_broad",
    "step_broad",
    "PolyFunc",
    "MultiPeakFunction",
    "FermiEdge2dFunc",
]

from erlab.analysis.fit.functions.general import (
    TINY,
    do_convolve,
    do_convolve_y,
    gaussian_wh,
    lorentzian_wh,
    fermi_dirac,
    fermi_dirac_linbkg,
    fermi_dirac_linbkg_broad,
    step_linbkg_broad,
    step_broad,
)

from erlab.analysis.fit.functions.dynamic import (
    PolyFunc,
    MultiPeakFunction,
    FermiEdge2dFunc,
)
