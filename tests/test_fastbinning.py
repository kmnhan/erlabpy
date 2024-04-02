import numpy as np
from erlab.interactive.imagetool.fastbinning import nanmean_funcs, fast_nanmean


def test_fast_nanmean():
    for nd, funcs in nanmean_funcs.items():
        x = np.random.RandomState(42).randn(*((10,) * nd))
        for axis, func in funcs.items():
            if isinstance(axis, frozenset):
                axis = tuple(axis)
            if not np.allclose(np.nanmean(x, axis), fast_nanmean(x, axis)):
                raise AssertionError(
                    f"fast_nanmean failed for {nd}D array "
                    f"with axis {axis} using {func}."
                )
