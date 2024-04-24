from collections.abc import Callable

import erlab.analysis.kspace
import numpy as np

k_tot = np.array([1.0, 2.0, 3.0])


def _generate_funclist() -> list[tuple[Callable, Callable]]:
    funcs = []

    for kconv_func in (
        erlab.analysis.kspace._kconv_func_type1,
        erlab.analysis.kspace._kconv_func_type2,
    ):
        for delta, xi, xi0, beta0 in zip(
            [0, 90.0, -90.0],
            [0, 30.0, -30.0],
            [0.0, 10.0, -10.0],
            [0.0, 10.0, -10.0],
            strict=True,
        ):
            funcs.append(kconv_func(k_tot, delta, xi, xi0, beta0))
    for kconv_func in (
        erlab.analysis.kspace._kconv_func_type1_da,
        erlab.analysis.kspace._kconv_func_type2_da,
    ):
        for delta, chi, chi0, xi, xi0 in zip(
            [0, 90.0, -90.0],
            [0, 10.0, -30.0],
            [0.0, 10.0, -10.0],
            [0.0, 10.0, -10.0],
            [0.0, 10.0, -10.0],
            strict=True,
        ):
            funcs.append(kconv_func(k_tot, delta, chi, chi0, xi, xi0))
    return funcs


def test_transform():
    for forward_func, inverse_func in _generate_funclist():
        alpha, beta = inverse_func(*forward_func(10.0, 20.0))
        assert alpha.size == beta.size == k_tot.size
        assert np.allclose(alpha, 10.0)
        assert np.allclose(beta, 20.0)

        kx, ky = forward_func(*inverse_func(0.1, 0.2))
        assert kx.size == ky.size == k_tot.size
        assert np.allclose(kx, 0.1)
        assert np.allclose(ky, 0.2)
