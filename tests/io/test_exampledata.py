import numpy as np

from erlab.io.exampledata import generate_data, generate_data_angles, generate_gold_edge


def test_generate_data() -> None:
    data = generate_data((3, 3, 3), seed=1)

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [0.48626864, 1.05399704, 0.25573808],
                    [0.12365866, 0.30740126, 0.07550997],
                    [0.12574204, 0.46739134, 0.1179754],
                ],
                [
                    [1.05542634, 0.7415651, 0.14367356],
                    [0.40084005, 0.53844017, 0.12289731],
                    [0.92840873, 1.92327964, 0.46456906],
                ],
                [
                    [0.26281194, 0.14510285, 0.02525125],
                    [0.53795665, 0.21633611, 0.03021987],
                    [2.39963975, 0.92558645, 0.12431274],
                ],
            ]
        ),
    )

    np.testing.assert_allclose(data.kx.values, np.array([-0.89, 0.0, 0.89]))
    np.testing.assert_allclose(data.ky.values, np.array([-0.89, 0.0, 0.89]))


def test_generate_data_angles() -> None:
    data = generate_data_angles(
        (3, 3, 3),
        hv=50.0,
        configuration=1,
        temp=20.0,
        seed=1,
        assign_attributes=True,
    )

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [2890.86015633, 1377.42170112, 221.92375664],
                    [526.0630528, 378.2380442, 76.39262082],
                    [2840.30481356, 1373.08414279, 225.41964774],
                ],
                [
                    [1693.41483286, 1289.97192121, 259.3909047],
                    [739.2855479, 447.16504001, 84.41399777],
                    [1707.06105363, 1291.77555895, 259.8300259],
                ],
                [
                    [2829.98953077, 1409.20187748, 232.82531889],
                    [531.0130294, 382.5037929, 77.92487419],
                    [2895.19162554, 1396.39378849, 224.93934341],
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
