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


def correct_with_edge(*args, **kwargs):
    from erlab.analysis.gold import correct_with_edge

    warnings.warn(
        "importing as erlab.analysis.correct_with_edge is deprecated, "
        "use erlab.analysis.gold.correct_with_edge instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return correct_with_edge(*args, **kwargs)


def quick_resolution(*args, **kwargs):
    from erlab.analysis.gold import quick_resolution

    warnings.warn(
        "importing as erlab.analysis.quick_resolution is deprecated, "
        "use erlab.analysis.gold.quick_resolution instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return quick_resolution(*args, **kwargs)


def slice_along_path(*args, **kwargs):
    from erlab.analysis.interpolate import slice_along_path

    warnings.warn(
        "importing as erlab.analysis.slice_along_path is deprecated, "
        "use erlab.analysis.interpolate.slice_along_path instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return slice_along_path(*args, **kwargs)
