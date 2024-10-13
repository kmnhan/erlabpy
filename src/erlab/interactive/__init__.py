"""
Interactive plotting based on Qt and pyqtgraph.

.. currentmodule:: erlab.interactive

Interactive tools
=================

.. autosummary::
   :toctree: generated

   imagetool
   bzplot
   colors
   curvefittingtool
   fermiedge
   kspace
   masktool
   derivative
   utils

"""

__all__ = ["dtool", "goldtool", "itool", "ktool"]


try:
    import qtpy  # noqa: F401
except ImportError as e:
    raise ImportError(
        "A Qt binding is required for interactive tools. "
        "Please install one of the following packages:\n"
        "  - PySide6: 'pip install PySide6'\n"
        "  - PyQt6: 'pip install PyQt6'\n"
        "For more information, visit the official documentation of these packages."
    ) from e
from erlab.interactive.derivative import dtool
from erlab.interactive.fermiedge import goldtool
from erlab.interactive.imagetool import itool
from erlab.interactive.kspace import ktool
