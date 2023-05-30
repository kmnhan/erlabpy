"""
================================================
Characterization (:mod:`erlab.characterization`)
================================================

.. currentmodule:: erlab.characterization

Data import and analysis for characterization experiments.

Modules
=======

.. autosummary::
   :toctree: generated
   
   xrd
   resistance

"""

__all__ = ["load_resistance_physlab", "load_xrd_itx"]

from erlab.characterization.resistance import load_resistance_physlab
from erlab.characterization.xrd import load_xrd_itx
