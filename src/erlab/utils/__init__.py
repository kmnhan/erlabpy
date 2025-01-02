"""
Generic utilities used in various parts of the package.

Most of the functions in this module are used internally and are not intended to be used
directly by the user.

.. currentmodule:: erlab.utils

Modules
=======

.. autosummary::
   :toctree: generated

   array
   parallel
   formatting
   misc

"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
