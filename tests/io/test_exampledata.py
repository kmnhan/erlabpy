import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.io.exampledata import (
    generate_data,
    generate_data_angles,
    generate_data_dirac,
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


def test_generate_data_angles_geometry() -> None:
    data = generate_data_angles(
        (7, 5, 3),
        noise=False,
        extended=False,
        normal_emission=(2.0, -1.5),
        delta_offset=-4.0,
        assign_attributes=True,
    )

    assert dict(data.kspace.offsets) == {
        "delta": -4.0,
        "xi": -2.0,
        "beta": -1.5,
    }
    np.testing.assert_allclose(data.kspace._forward_func(2.0, -1.5), 0.0, atol=1e-14)


def test_generate_data_angles_geometry_defaults_are_compatible() -> None:
    default = generate_data_angles((4, 3, 5), seed=2, assign_attributes=True)
    explicit = generate_data_angles(
        (4, 3, 5),
        seed=2,
        assign_attributes=True,
        normal_emission=(0.0, 0.0),
        delta_offset=0.0,
        band_rotation=0.0,
    )

    xr.testing.assert_identical(default, explicit)


def test_generate_data_angles_rotates_band_structure(monkeypatch) -> None:
    original = erlab.io.exampledata._band
    band_points: list[np.ndarray] = []

    def record_band_points(kvec, *args):
        band_points.append(kvec.copy())
        return original(kvec, *args)

    monkeypatch.setattr(erlab.io.exampledata, "_band", record_band_points)
    kwargs = {
        "shape": (4, 3, 2),
        "noise": False,
        "extended": False,
        "normal_emission": (2.0, -1.5),
        "delta_offset": -4.0,
    }
    generate_data_angles(**kwargs)
    generate_data_angles(**kwargs, band_rotation=-30.0)

    theta = np.deg2rad(30.0)
    rotation = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    np.testing.assert_allclose(band_points[1], band_points[0] @ rotation)


@pytest.mark.parametrize(("extended", "expected_calls"), [(False, 1), (True, 2)])
def test_generate_data_angles_geometry_reaches_each_mapping_path(
    monkeypatch: pytest.MonkeyPatch, extended: bool, expected_calls: int
) -> None:
    original = erlab.analysis.kspace.get_kconv_forward
    calls: list[dict[str, float]] = []

    def get_forward(configuration):
        forward = original(configuration)

        def record(*args, **kwargs):
            calls.append(kwargs.copy())
            return forward(*args, **kwargs)

        return record

    monkeypatch.setattr(erlab.analysis.kspace, "get_kconv_forward", get_forward)

    data = generate_data_angles(
        (5, 4, 3),
        noise=False,
        extended=extended,
        normal_emission=(2.0, -1.5),
        delta_offset=-4.0,
    )

    assert len(calls) == expected_calls
    assert all(
        call
        == {
            "delta": -4.0,
            "xi": 0.0,
            "xi0": -2.0,
            "beta0": -1.5,
        }
        for call in calls
    )
    np.testing.assert_allclose(data.alpha, np.linspace(-15.0, 15.0, 5))
    np.testing.assert_allclose(data.beta, np.linspace(-15.0, 15.0, 4))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"normal_emission": (0.0,)}, "finite 2-tuple"),
        ({"normal_emission": (0.0, np.inf)}, "finite 2-tuple"),
        ({"normal_emission": iter((0.0, 0.0))}, "finite 2-tuple"),
        ({"delta_offset": [0.0]}, "finite scalar"),
        ({"delta_offset": np.nan}, "finite scalar"),
        ({"band_rotation": [0.0]}, "finite scalar"),
        ({"band_rotation": np.nan}, "finite scalar"),
    ],
)
def test_generate_data_angles_invalid_geometry(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        generate_data_angles((3, 3, 3), **kwargs)


def test_generate_gold_edge() -> None:
    data = generate_gold_edge((3, 5), seed=1)

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [102.10008336, 138.5670346, 107.91550941],
                [96.93029016, 123.40696181, 81.97504237],
                [78.83052462, 93.82780779, 65.01473936],
                [65.03657059, 73.73499564, 55.10391992],
                [14.43371294, 16.35223181, 11.77057561],
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


def test_generate_data_dirac() -> None:
    data = generate_data_dirac(
        (3, 3, 3), noise=False, temp=0, seed=1, assign_attributes=True
    )

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [6656.18118342, 1343.59264981, 1426.80072988],
                    [1656.67581261, 4968.02848953, 4148.18055093],
                    [6656.18118342, 1343.59264981, 1426.80072988],
                ],
                [
                    [2126.69563286, 6398.92197193, 3497.00148285],
                    [736.99071449, 5380.40885929, 9984.87072098],
                    [2126.69563286, 6398.92197193, 3497.00148285],
                ],
                [
                    [11165.50429819, 1975.45108131, 1055.44064151],
                    [2596.71545311, 7829.81545433, 2845.82241477],
                    [11165.50429819, 1975.45108131, 1055.44064151],
                ],
            ]
        ),
    )
    np.testing.assert_allclose(data.alpha.values, np.array([-15.0, 0.0, 15.0]))
    np.testing.assert_allclose(data.beta.values, np.array([-15.0, 0.0, 15.0]))
    np.testing.assert_allclose(data.eV.values, np.array([-0.45, -0.165, 0.12]))

    assert data.attrs["dirac_branch"] == "both"
    assert data.attrs["dirac_spin"] == "integrated"
    assert data.attrs["dirac_velocity"] == 0.3
    assert data.attrs["dirac_beta_coeff"] == 0.6


def test_generate_data_dirac_spin_projection() -> None:
    data = generate_data_dirac(
        (3, 3, 3), noise=False, temp=0, spin="up", branch="upper"
    )

    np.testing.assert_allclose(
        data.values,
        np.array(
            [
                [
                    [46.67909038, 109.71512571, 491.81472787],
                    [70.40086871, 194.25547082, 1603.95759444],
                    [46.67909038, 109.71512571, 491.81472787],
                ],
                [
                    [62.92411626, 173.60633207, 1433.38152019],
                    [184.32267862, 1345.17721482, 2496.29268024],
                    [62.92411626, 173.60633207, 1433.38152019],
                ],
                [
                    [46.67909038, 109.71512571, 491.81472787],
                    [70.40086871, 194.25547082, 1603.95759444],
                    [46.67909038, 109.71512571, 491.81472787],
                ],
            ]
        ),
    )
