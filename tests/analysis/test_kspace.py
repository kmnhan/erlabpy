from collections.abc import Callable

import numpy as np

import erlab.analysis

kinetic = np.array([1.0, 2.0, 3.0])


def _generate_funclist() -> list[tuple[Callable, Callable]]:
    funcs = []

    for delta, xi, xi0, beta0 in zip(
        [0, 90.0, -90.0],
        [0, 30.0, -30.0],
        [0.0, 10.0, -10.0],
        [0.0, 10.0, -10.0],
        strict=True,
    ):
        for config in (1, 2):
            funcs.append(  # noqa: PERF401
                erlab.analysis.kspace.get_kconv_func(
                    kinetic,
                    config,
                    {"delta": delta, "xi": xi, "xi0": xi0, "beta0": beta0},
                )
            )
    for delta, chi, chi0, xi, xi0 in zip(
        [0, 90.0, -90.0],
        [0, 10.0, -30.0],
        [0.0, 10.0, -10.0],
        [0.0, 10.0, -10.0],
        [0.0, 10.0, -10.0],
        strict=True,
    ):
        for config in (3, 4):
            funcs.append(  # noqa: PERF401
                erlab.analysis.kspace.get_kconv_func(
                    kinetic,
                    config,
                    {"delta": delta, "chi": chi, "chi0": chi0, "xi": xi, "xi0": xi0},
                )
            )

    return funcs


def test_transform() -> None:
    for forward_func, inverse_func in _generate_funclist():
        alpha, beta = inverse_func(*forward_func(10.0, 20.0))
        assert alpha.size == beta.size == kinetic.size
        assert np.allclose(alpha, 10.0)
        assert np.allclose(beta, 20.0)

        kx, ky = forward_func(*inverse_func(0.1, 0.2))
        assert kx.size == ky.size == kinetic.size
        assert np.allclose(kx, 0.1)
        assert np.allclose(ky, 0.2)
