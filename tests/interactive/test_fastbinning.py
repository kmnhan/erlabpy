import numpy as np

from erlab.interactive.imagetool.fastbinning import NANMEAN_FUNCS, fast_nanmean


def test_fast_nanmean() -> None:
    for nd, funcs in NANMEAN_FUNCS.items():
        x = np.random.RandomState(42).randn(*((10,) * nd))
        x_double = x.astype(np.float64)
        x_single = x.astype(np.float32)
        for axis, func in funcs.items():
            if isinstance(axis, frozenset):
                axis = tuple(axis)
            if not np.allclose(
                np.nanmean(x_double, axis), fast_nanmean(x_double, axis)
            ):
                raise AssertionError(
                    f"fast_nanmean failed for {nd}D float64 array "
                    f"with axis {axis} using {func}."
                )
            if not np.allclose(
                np.nanmean(x_single.astype(np.float64), axis).astype(np.float32),
                fast_nanmean(x_single, axis),
            ):
                raise AssertionError(
                    f"fast_nanmean failed for {nd}D float32 array "
                    f"with axis {axis} using {func}."
                )
