import numpy as np

from erlab.io.exampledata import (
    generate_data,
    generate_data_angles,
    generate_gold_edge,
    generate_hvdep_cuts,
)


def test_generate_data() -> None:
    data = generate_data((3, 3, 3), seed=1)

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [8.33879501, 3.62012935, 0.5394082],
                    [4.54361884, 1.9473626, 0.28721561],
                    [12.09026517, 5.59104172, 0.87315513],
                ],
                [
                    [7.40116677, 5.88533774, 1.18981106],
                    [5.36249153, 3.00372148, 0.52667583],
                    [10.85406545, 6.67713352, 1.22498652],
                ],
                [
                    [7.01310157, 2.65885305, 0.35101],
                    [4.15030691, 2.12130577, 0.35349177],
                    [10.6496747, 4.17601482, 0.56985991],
                ],
            ]
        ),
    )

    np.testing.assert_allclose(data.kx.values, np.array([-0.89, 0.0, 0.89]))
    np.testing.assert_allclose(data.ky.values, np.array([-0.89, 0.0, 0.89]))


def test_generate_data_angles() -> None:
    data = generate_data_angles(
        (3, 3, 3), hv=50.0, configuration=1, temp=20.0, seed=1, assign_attributes=True
    )

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [106.56181151, 49.07320594, 7.9050182],
                    [17.72867797, 13.46779154, 2.68245659],
                    [102.1657067, 47.50953682, 7.44858551],
                ],
                [
                    [60.45474898, 45.30683692, 10.02450486],
                    [26.23142958, 16.75257948, 3.12430078],
                    [67.10235524, 52.78666754, 10.63503772],
                ],
                [
                    [106.16327022, 52.63703719, 8.87356619],
                    [19.51693294, 15.39220884, 3.10362227],
                    [112.04303551, 50.55278567, 7.75628928],
                ],
            ]
        ),
    )
    np.testing.assert_allclose(data.alpha.values, np.array([-15.0, 0.0, 15.0]))
    np.testing.assert_allclose(data.beta.values, np.array([-15.0, 0.0, 15.0]))

    np.testing.assert_allclose(data.xi.values, 0.0)
    np.testing.assert_allclose(data.delta.values, 0.0)
    np.testing.assert_allclose(data.hv.values, 50.0)

    assert data.attrs["sample_temp"] == 20.0
    assert data.attrs["configuration"] == 1


def test_generate_gold_edge() -> None:
    data = generate_gold_edge((3, 5), seed=1)

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [8.16681531, 11.46644012, 4.87101708],
                [7.71234876, 11.13652632, 7.14514618],
                [8.69495264, 9.415107, 6.49044818],
                [8.71305841, 6.36380146, 5.56047722],
                [1.86721051, 1.23067337, 1.16597745],
            ]
        ),
    )

    np.testing.assert_allclose(data.eV.values, np.array([-1.3, -0.9, -0.5, -0.1, 0.3]))
    np.testing.assert_allclose(data.alpha.values, np.array([-15.0, 0.0, 15.0]))


def test_generate_hvdep_cuts() -> None:
    data = generate_hvdep_cuts((3, 3, 3), noise=False, temp=0, hvrange=(40, 60))

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [79.90509526, 46.07895401, 50.66075799],
                    [30.47582358, 29.52025701, 34.92976538],
                    [51.82761664, 43.65856414, 47.96927155],
                ],
                [
                    [100.26801478, 143.17916656, 195.53833693],
                    [35.33446342, 50.39667113, 68.76074904],
                    [18.00410008, 25.62919158, 34.91833759],
                ],
                [
                    [79.90509526, 46.07895401, 50.66075799],
                    [30.47582358, 29.52025701, 34.92976538],
                    [51.82761664, 43.65856414, 47.96927155],
                ],
            ]
        ),
    )

    np.testing.assert_allclose(data.eV.values, np.array([-0.45, -0.165, 0.12]))
    np.testing.assert_allclose(data.hv.values, np.array([40, 50, 60]))
    np.testing.assert_allclose(data.alpha.values, np.array([-15.0, 0.0, 15.0]))
