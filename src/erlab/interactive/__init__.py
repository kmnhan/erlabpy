"""
Interactive plotting based on Qt and pyqtgraph

.. currentmodule:: erlab.interactive

Interactive tools
=================

.. autosummary::
   :toctree: generated

   imagetool
   bzplot
   colors
   curvefittingtool
   exampledata
   fermiedge
   kspace
   masktool
   derivative
   utilities

"""

__all__ = ["goldtool", "itool", "ktool", "dtool"]

import importlib

from erlab.interactive.fermiedge import goldtool
from erlab.interactive.imagetool import itool
from erlab.interactive.kspace import ktool

if importlib.util.find_spec("arpes"):
    from erlab.interactive.derivative import dtool
