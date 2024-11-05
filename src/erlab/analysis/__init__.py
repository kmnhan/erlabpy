"""
Various functions for data analysis.

.. currentmodule:: erlab.analysis

Modules
=======

.. autosummary::
   :toctree: generated

   fit
   mask
   correlation
   gold
   image
   interpolate
   kspace
   transform

"""

import warnings

from erlab.analysis import fit, gold, image, interpolate, mask, transform  # noqa: F401


def slice_along_path(*args, **kwargs):
    from erlab.analysis.interpolate import slice_along_path

    warnings.warn(
        "importing as erlab.analysis.slice_along_path is deprecated, "
        "use erlab.analysis.interpolate.slice_along_path instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return slice_along_path(*args, **kwargs)
