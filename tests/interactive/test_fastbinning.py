import numpy as np

from erlab.interactive.imagetool.fastbinning import NANMEAN_FUNCS, fast_nanmean


def test_fast_nanmean_signature() -> None:
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
                rtol=1e-7,
                atol=0,
            ):
                raise AssertionError(
                    f"fast_nanmean failed for {nd}D float32 array "
                    f"with axis {axis} using {func}."
                )


def test_fast_nanmean_5d() -> None:
    x = np.random.RandomState(42).randn(3, 4, 5, 6, 7).astype(np.float64)
    # Test with different axes
    for axis in [0, 1, 2, 3, (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        result = fast_nanmean(x, axis=axis)
        expected = np.nanmean(x, axis=axis)
        if not np.allclose(result, expected, rtol=1e-7, atol=0):
            raise AssertionError(f"fast_nanmean failed for axis {axis} for 5D array")
