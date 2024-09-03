import numpy as np

from erlab.io.exampledata import generate_data, generate_data_angles, generate_gold_edge


def test_generate_data():
    data = generate_data((3, 3, 3), seed=1)

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [9010277.53158856, 4458397.04074018, 809908.96443774],
                    [4698266.35922851, 2578636.07354791, 528809.26472864],
                    [9008422.18238725, 4456859.98787036, 809438.41409181],
                ],
                [
                    [5463001.22757439, 3986865.31769739, 864834.81723129],
                    [3594377.58855508, 2508454.42140497, 566255.6149856],
                    [5460569.85001597, 3984435.69921373, 864115.77259181],
                ],
                [
                    [9012950.87657523, 4459198.12053694, 809868.93308102],
                    [4700173.71802403, 2577709.04540753, 528257.25001757],
                    [9012753.85992549, 4458417.57349542, 809627.4950456],
                ],
            ]
        ),
    )

    np.testing.assert_allclose(data.kx.values, np.array([-0.89, 0.0, 0.89]))
    np.testing.assert_allclose(data.ky.values, np.array([-0.89, 0.0, 0.89]))


def test_generate_data_angles():
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
                    [10584103.6542194, 5093835.27779442, 896123.30180953],
                    [1709763.89081559, 1250221.69563657, 324333.8997436],
                    [10581859.86894855, 5091979.3893089, 895557.22568712],
                ],
                [
                    [6101271.24393742, 4502715.52572828, 966803.28944403],
                    [2240475.87672943, 1433784.37846328, 346174.48478143],
                    [6098365.34359401, 4499774.3417057, 965935.18855541],
                ],
                [
                    [10587162.63852743, 5095269.99906888, 896271.06947063],
                    [1710576.94066684, 1247915.58065668, 323377.25250659],
                    [10586937.5332778, 5094328.47440016, 895980.63452237],
                ],
            ]
        ),
    )
    np.testing.assert_allclose(data.alpha.values, np.array([-15.0, 0.0, 15.0]))
    np.testing.assert_allclose(data.beta.values, np.array([-15.0, 0.0, 15.0]))

    np.testing.assert_allclose(data.xi.values, 0.0)
    np.testing.assert_allclose(data.delta.values, 0.0)
    np.testing.assert_allclose(data.hv.values, 50.0)

    assert data.attrs["temp_sample"] == 20.0
    assert data.attrs["configuration"] == 1


def test_generate_gold_edge():
    data = generate_gold_edge(nx=3, ny=5, seed=1)

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [88861.71360087, 120239.1523542, 89026.19170356],
                [79988.11645269, 107963.33789627, 79549.67219177],
                [68850.81086526, 92741.26836792, 68413.28599851],
                [50556.8598757, 68031.55342743, 50174.42004713],
                [10742.14847358, 14218.90842119, 10640.56032391],
            ]
        ),
    )

    np.testing.assert_allclose(data.eV.values, np.array([-1.3, -0.9, -0.5, -0.1, 0.3]))
    np.testing.assert_allclose(data.alpha.values, np.array([-15.0, 0.0, 15.0]))
