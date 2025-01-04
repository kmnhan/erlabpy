"""Utilities for curve fitting.

For examples, see the :doc:`User Guide <../user-guide/curve-fitting>`.

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

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
