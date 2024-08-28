import warnings

import xarray as xr


def shift(*args, **kwargs) -> xr.DataArray:
    from erlab.analysis.transform import shift as shift_func

    warnings.warn(
        "erlab.analysis.utils.shift is deprecated, "
        "use erlab.analysis.gold.correct_with_edge instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return shift_func(*args, **kwargs)


def correct_with_edge(*args, **kwargs):
    from erlab.analysis.gold import correct_with_edge

    warnings.warn(
        "erlab.analysis.utils.correct_with_edge is deprecated, "
        "use erlab.analysis.gold.correct_with_edge instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return correct_with_edge(*args, **kwargs)
