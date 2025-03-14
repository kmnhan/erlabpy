"""
Generic utilities used in various parts of the package.

Most of the functions in this module are used internally, and are not likely to be used
directly when conducting data analysis. Advanced users may find some of the functions in
this module useful for building custom scripts or extending the functionality of this
package.

.. currentmodule:: erlab.utils

.. rubric:: Modules

.. autosummary::
   :toctree: generated

   array
   parallel
   formatting
   misc

"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
