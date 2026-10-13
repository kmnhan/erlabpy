import dataclasses

import numpy as np
import pytest

import erlab.analysis.kspace
import erlab.constants
from erlab.interactive.angle_calculator._calculation import (
    Geometry,
    angles_to_momentum,
    emission_radius,
    momentum_to_angles,
)
from erlab.interactive.angle_calculator._low_energy import (
    momentum_to_slopes,
    slopes_to_momentum,
)


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
def test_standard_geometry_matches_erlab(configuration):
    geometry = Geometry(
        photon_energy=40.0,
        configuration=configuration,
        polar=7.0,
        tilt=-8.0,
        azimuth=23.0,
    )
    alpha, beta = np.meshgrid([-8.0, 0.0, 6.0], [-4.0, 2.0])
    params = {"delta": geometry.azimuth}
    if configuration in (1, 2):
        params["xi"] = geometry.tilt if configuration == 1 else geometry.polar
    else:
        params.update(chi=geometry.polar, xi=geometry.tilt)
    expected = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha,
        beta,
        40.0 - geometry.analyzer_work_function,
        **params,
    )
    result = angles_to_momentum(alpha, beta, geometry)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(
        momentum_to_angles(*result, geometry),
        erlab.analysis.kspace.get_kconv_inverse(configuration)(
            *result, None, 40.0 - geometry.analyzer_work_function, **params
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        momentum_to_angles(*result, geometry), [alpha, beta], atol=1e-12
    )


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
def test_sample_hemisphere_mask(configuration):
    geometry = Geometry(
        photon_energy=100.0, polar=150.0, tilt=150.0, configuration=configuration
    )
    if configuration == 1 or configuration == 2:
        alpha, beta = 0.0, 0.0
    else:
        # One 180-degree mechanical rotation reverses the sample normal.
        geometry = dataclasses.replace(geometry, polar=180.0, tilt=0.0)
        alpha, beta = 0.0, 0.0
    assert np.isnan(angles_to_momentum(alpha, beta, geometry)).all()


@pytest.mark.parametrize("configuration", [3, 4])
def test_inverse_masks_rays_behind_analyzer(configuration):
    geometry = Geometry(photon_energy=40, polar=80, configuration=configuration)
    assert np.isnan(momentum_to_angles(0.0, 0.6, geometry)).all()
    np.testing.assert_allclose(
        angles_to_momentum(*momentum_to_angles(0.0, 0.5, geometry), geometry),
        [0.0, 0.5],
        atol=1e-12,
    )


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("polar", "tilt"), [(80.0, 0.0), (-80.0, 80.0), (150.0, 150.0)]
)
def test_inverse_physical_consistency_at_extreme_geometry(configuration, polar, tilt):
    geometry = Geometry(
        photon_energy=40.0,
        configuration=configuration,
        polar=polar,
        tilt=tilt,
    )
    kx, ky = np.meshgrid(np.linspace(-2.0, 2.0, 21), np.linspace(-2.0, 2.0, 21))
    alpha, beta = momentum_to_angles(kx, ky, geometry)
    valid = np.isfinite(alpha) & np.isfinite(beta)
    np.testing.assert_allclose(
        angles_to_momentum(alpha[valid], beta[valid], geometry),
        [kx[valid], ky[valid]],
        atol=1e-12,
    )
    if configuration == 2 and polar == 150.0:
        assert np.isnan(momentum_to_angles(0.0, 0.0, geometry)).all()


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
def test_low_photon_energy_matches_public_conversion(configuration):
    geometry = Geometry(
        photon_energy=6.2,
        binding_energy=-0.2,
        analyzer_work_function=4.65,
        configuration=configuration,
    )
    result = angles_to_momentum(30.0, 0.0, geometry)
    np.testing.assert_allclose(
        result,
        erlab.analysis.kspace.get_kconv_forward(configuration)(30.0, 0.0, 1.35),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        momentum_to_angles(*result, geometry), [30.0, 0.0], atol=1e-12
    )


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
def test_thresholds_and_invalid_points(configuration):
    geometry = Geometry(photon_energy=6.0, configuration=configuration)
    assert np.isnan(momentum_to_angles([1.0, np.nan, np.inf], 0.0, geometry)).all()
    assert np.isnan(angles_to_momentum([100.0, np.nan, np.inf], 0.0, geometry)).all()
    for work_function in (6.0, 7.0, np.nan, np.inf):
        blocked = dataclasses.replace(geometry, analyzer_work_function=work_function)
        assert np.isnan(angles_to_momentum(0.0, 0.0, blocked)).all()
        assert np.isnan(momentum_to_angles(0.0, 0.0, blocked)).all()


@pytest.mark.parametrize("energies", [(0.1, 1.35), (1.35, 0.1), (1.35, 1.35)])
def test_private_position_limit_roundtrip(energies):
    slope_x, slope_y = np.broadcast_arrays(
        np.array([-0.1, 0.0, 0.12])[:, None], np.array([-0.08, 0.0, 0.09])
    )
    momentum = slopes_to_momentum(slope_x, slope_y, *energies)
    np.testing.assert_allclose(
        momentum_to_slopes(*momentum, *energies), [slope_x, slope_y], atol=1e-12
    )
    if energies[0] == energies[1]:
        scale = erlab.constants.rel_kconv * np.sqrt(energies[0])
        np.testing.assert_allclose(
            momentum,
            np.array([slope_x, slope_y]) * scale / np.sqrt(1 + slope_x**2 + slope_y**2),
            atol=1e-12,
        )


def test_private_position_limit_reference_and_common_ray():
    slope = np.tan(np.deg2rad(15.0))
    # Components of one position ray, including the deflector rotation-vector ray.
    kx, ky = slopes_to_momentum(-0.6 * slope, 0.8 * slope, 0.1, 1.35)
    assert np.hypot(kx, ky) == pytest.approx(0.096169120875, abs=1e-12)
    assert kx / ky == pytest.approx(-0.75)
    np.testing.assert_allclose(momentum_to_slopes(0.0, 0.0, 0.1, 1.35), [0.0, 0.0])


@pytest.mark.parametrize("energies", [(0.1, 1.35), (1.35, 0.1)])
def test_private_position_limit_rejects_backward_branch(energies):
    sample_energy, analyzer_energy = energies
    slope = np.tan(np.deg2rad(30.0))
    g = (analyzer_energy - sample_energy) / (2 * sample_energy)
    # Real quadratic roots alone do not ensure forward velocities at both plates.
    assert 1 + 2 * g - g**2 * slope**2 > 0
    assert np.isnan(slopes_to_momentum(slope, 0.0, *energies)).all()
    assert np.isfinite(
        slopes_to_momentum(np.tan(np.deg2rad(15.0)), 0.0, *energies)
    ).all()
    radius = erlab.constants.rel_kconv * np.sqrt(min(energies))
    assert np.isnan(momentum_to_slopes(radius * 1.01, 0.0, *energies)).all()


@pytest.mark.parametrize("energy", [0.0, -0.1, np.nan, np.inf])
@pytest.mark.parametrize("sample", [False, True])
def test_private_position_limit_invalid_energies(energy, sample):
    energies = (energy, 1.35) if sample else (1.35, energy)
    assert np.isnan(slopes_to_momentum([0.0, 0.1], 0.0, *energies)).all()
    assert np.isnan(momentum_to_slopes([0.0, 0.1], 0.0, *energies)).all()


def test_private_position_limit_invalid_coordinates():
    assert np.isnan(slopes_to_momentum([np.nan, np.inf, 10.0], 0.0, 0.1, 1.35)).all()
    assert np.isnan(momentum_to_slopes([np.nan, np.inf], 0.0, 0.1, 1.35)).all()


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
@pytest.mark.parametrize("work_functions", [(5.9, 4.65), (4.65, 5.9), (4.65, 4.65)])
def test_corrected_geometry_roundtrip(configuration, work_functions):
    geometry = Geometry(
        photon_energy=6.0,
        configuration=configuration,
        sample_work_function=work_functions[0],
        analyzer_work_function=work_functions[1],
        low_energy_correction=True,
        azimuth=37.0,
    )
    alpha = np.array([-8.0, 0.0, 7.0])[:, None]
    beta = np.array([-5.0, 0.0, 4.0]) if configuration in (3, 4) else np.array([0.0])
    momentum = angles_to_momentum(alpha, beta, geometry)
    np.testing.assert_allclose(
        momentum_to_angles(*momentum, geometry),
        np.broadcast_arrays(alpha, beta),
        atol=1e-12,
    )
    if work_functions[0] == work_functions[1]:
        np.testing.assert_allclose(
            momentum,
            angles_to_momentum(
                alpha, beta, dataclasses.replace(geometry, low_energy_correction=False)
            ),
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("configuration", "field"),
    [(1, "tilt"), (2, "polar"), (3, "polar"), (3, "tilt"), (4, "polar"), (4, "tilt")],
)
def test_correction_rejects_non_normal_geometry(configuration, field):
    geometry = dataclasses.replace(
        Geometry(configuration=configuration, low_energy_correction=True),
        **{field: 1.0},
    )
    with pytest.raises(ValueError, match="normal emission geometry"):
        angles_to_momentum(0.0, 0.0, geometry)
    with pytest.raises(ValueError, match="normal emission geometry"):
        momentum_to_angles(0.0, 0.0, geometry)


@pytest.mark.parametrize(("configuration", "field"), [(1, "polar"), (2, "tilt")])
def test_correction_ignores_inactive_mechanical_angle(configuration, field):
    geometry = dataclasses.replace(
        Geometry(configuration=configuration, low_energy_correction=True),
        **{field: 20.0},
    )
    momentum = angles_to_momentum(5.0, 0.0, geometry)
    np.testing.assert_allclose(
        momentum_to_angles(*momentum, geometry), [5.0, 0.0], atol=1e-12
    )
    with pytest.raises(ValueError, match="zero mechanical beta"):
        angles_to_momentum(5.0, 1.0, geometry)
    kx, ky = (0.0, 0.01) if configuration == 1 else (0.01, 0.0)
    assert np.isnan(momentum_to_angles(kx, ky, geometry)).all()


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
def test_standard_geometry_is_independent_of_sample_work_function(configuration):
    geometry = Geometry(photon_energy=6.0, configuration=configuration)
    expected = angles_to_momentum(30.0, 0.0, geometry)
    for work_function in (5.9, 6.0, 7.0, np.nan):
        changed = dataclasses.replace(geometry, sample_work_function=work_function)
        np.testing.assert_allclose(angles_to_momentum(30.0, 0.0, changed), expected)
        np.testing.assert_allclose(
            momentum_to_angles(*expected, changed), [30.0, 0.0], atol=1e-12
        )


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
@pytest.mark.parametrize("work_functions", [(5.9, 4.65), (4.65, 5.9)])
def test_corrected_geometry_masks_physical_domain(configuration, work_functions):
    geometry = Geometry(
        photon_energy=6.0,
        configuration=configuration,
        sample_work_function=work_functions[0],
        analyzer_work_function=work_functions[1],
        low_energy_correction=True,
    )
    assert np.isnan(angles_to_momentum(30.0, 0.0, geometry)).all()
    assert np.isfinite(angles_to_momentum(15.0, 0.0, geometry)).all()
    for field in ("sample_work_function", "analyzer_work_function"):
        blocked = dataclasses.replace(geometry, **{field: 6.0})
        assert np.isnan(angles_to_momentum(0.0, 0.0, blocked)).all()
        assert np.isnan(momentum_to_angles(0.0, 0.0, blocked)).all()


@pytest.mark.parametrize(
    ("work_functions", "energy"),
    [((5.9, 4.65), 0.1), ((4.65, 5.9), 0.1), ((4.65, 4.65), 1.35), ((4.3, 6.0), 0.0)],
)
def test_emission_radius_includes_analyzer(work_functions, energy):
    geometry = Geometry(
        photon_energy=6.2,
        binding_energy=-0.2,
        sample_work_function=work_functions[0],
        analyzer_work_function=work_functions[1],
    )
    assert emission_radius(geometry) == pytest.approx(
        erlab.constants.rel_kconv * np.sqrt(6.0 - work_functions[0])
    )
    assert emission_radius(geometry, include_analyzer=True) == pytest.approx(
        erlab.constants.rel_kconv * np.sqrt(energy), abs=1e-12
    )
    for field in ("sample_work_function", "analyzer_work_function"):
        for work_function in (7.0, np.nan, np.inf):
            blocked = dataclasses.replace(geometry, **{field: work_function})
            assert np.isnan(emission_radius(blocked, include_analyzer=True))


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
@pytest.mark.parametrize("corrected", [False, True])
def test_three_dimensional_forward_shared_physics(configuration, corrected):
    from erlab.interactive.angle_calculator._calculation import angles_to_momentum3d

    geometry = Geometry(
        configuration=configuration,
        low_energy_correction=corrected,
        binding_energy=-0.3,
        inner_potential=12.5,
        sample_work_function=5.2,
        analyzer_work_function=4.5,
        azimuth=24.0,
        polar=0.0 if corrected else 13.0,
        tilt=0.0 if corrected else -7.0,
    )
    alpha = np.array([-12.0, 0.0, 8.0])[None, :]
    hv = np.array([6.0, 21.2, 90.0])[:, None]
    kx, ky, kz = angles_to_momentum3d(alpha, 0.0, geometry, hv)
    work_function = (
        geometry.sample_work_function if corrected else geometry.analyzer_work_function
    )
    np.testing.assert_allclose(
        kz,
        erlab.analysis.kspace.kz_func(
            hv + geometry.binding_energy - work_function,
            geometry.inner_potential,
            kx,
            ky,
        ),
    )
    np.testing.assert_allclose(
        erlab.analysis.kspace.hv_func(
            kx, ky, kz, geometry.inner_potential, work_function, geometry.binding_energy
        ),
        np.broadcast_to(hv, kx.shape),
    )
    for index, photon_energy in enumerate(hv[:, 0]):
        np.testing.assert_allclose(
            [kx[index], ky[index]],
            np.asarray(
                angles_to_momentum(
                    alpha[0],
                    0.0,
                    dataclasses.replace(geometry, photon_energy=photon_energy),
                )
            ),
        )


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize("work_functions", [(4.5, 5.2), (5.2, 4.5), (4.5, 4.5)])
def test_alpha_kz_surface_roundtrip(configuration, corrected, work_functions):
    from erlab.interactive.angle_calculator._calculation import (
        alpha_kz_surface,
        angles_to_momentum3d,
    )

    geometry = Geometry(
        configuration=configuration,
        low_energy_correction=corrected,
        binding_energy=-1.3,
        inner_potential=8.0,
        sample_work_function=work_functions[0],
        analyzer_work_function=work_functions[1],
        azimuth=31.0,
        polar=0.0 if corrected else 12.0,
        tilt=0.0 if corrected else -9.0,
    )
    alpha = np.array([-22.0, 0.0, 13.0])
    kz = np.array([1.8, 2.5, 4.0])
    beta = 0.0 if corrected and configuration in (1, 2) else 7.0
    surface, hv = alpha_kz_surface(alpha, kz, beta, geometry)
    assert surface.shape == (3, 3, 3)
    assert np.isfinite(surface).all()
    expected = np.stack(
        angles_to_momentum3d(alpha[None, :], beta, geometry, hv), axis=-1
    )
    np.testing.assert_allclose(surface, expected, atol=1e-12)
    np.testing.assert_allclose(surface[..., 2], np.broadcast_to(kz[:, None], (3, 3)))
    shifted = dataclasses.replace(
        geometry, binding_energy=geometry.binding_energy - 2.0
    )
    shifted_surface, shifted_hv = alpha_kz_surface(alpha, kz, beta, shifted)
    np.testing.assert_allclose(shifted_surface, surface)
    np.testing.assert_allclose(shifted_hv, hv + 2.0)


@pytest.mark.parametrize("corrected", [False, True])
def test_alpha_kz_surface_masks_thresholds(corrected):
    from erlab.interactive.angle_calculator._calculation import alpha_kz_surface

    geometry = Geometry(low_energy_correction=corrected)
    surface, hv = alpha_kz_surface(
        [-100.0, 0.0, 100.0], [-2.0, 0.0, 2.0], 0.0, geometry
    )
    assert np.isnan(surface[:2]).all()
    assert np.isnan(surface[:, [0, 2]]).all()
    assert np.isnan(hv[:2]).all()
    assert np.isnan(hv[:, [0, 2]]).all()
    assert np.isfinite(surface[2, 1]).all()


@pytest.mark.parametrize("corrected", [False, True])
def test_constant_energy_surface_uses_shared_kz(corrected):
    from erlab.interactive.angle_calculator._calculation import momentum_surface

    geometry = Geometry(
        photon_energy=6.2,
        binding_energy=-0.2,
        sample_work_function=5.5,
        analyzer_work_function=4.5,
        low_energy_correction=corrected,
    )
    surface = momentum_surface(
        [-10.0, -0.1, 0.0, 0.1, 10.0], [-0.1, 0.0, 0.1], geometry
    )
    assert np.isnan(surface[:, [0, 4]]).all()
    assert np.isfinite(surface[:, 1:4]).all()
    work_function = (
        geometry.sample_work_function if corrected else geometry.analyzer_work_function
    )
    np.testing.assert_allclose(
        surface[..., 2],
        erlab.analysis.kspace.kz_func(
            geometry.photon_energy + geometry.binding_energy - work_function,
            geometry.inner_potential,
            surface[..., 0],
            surface[..., 1],
        ),
    )
    blocked = dataclasses.replace(geometry, photon_energy=1.0)
    assert np.isnan(momentum_surface([0.0], [0.0], blocked)).all()


def test_three_dimensional_state_roundtrip():
    from erlab.interactive.angle_calculator._state import AngleCalculatorState

    state = AngleCalculatorState(
        lattice_a=4.123456789,
        lattice_b=5.234567891,
        lattice_c=7.345678912,
        lattice_alpha=80.123456789,
        lattice_beta=95.234567891,
        lattice_gamma=110.345678912,
        centering="I",
        lattice_rotation=12.0,
        hv_range=(6.0, 100.0),
        out_of_plane_visible=True,
        out_of_plane_range=((-10.0, 20.0), (1.5, 4.0)),
        geometry=Geometry(inner_potential=15.0),
    )
    assert AngleCalculatorState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.parametrize(
    "fields",
    [
        {"geometry": Geometry(inner_potential=-1.0)},
        {"geometry": Geometry(inner_potential=50.001)},
        {"geometry": Geometry(inner_potential=np.inf)},
        {"hv_range": (0.0, 10.0)},
        {"hv_range": (0.0001, 10.0)},
        {"hv_range": (4.999, 10.0)},
        {"hv_range": (20.0, 10.0)},
        {"hv_range": (20.0, np.inf)},
        {"hv_range": (20.0, 1000.001)},
        *(
            {f"lattice_{name}": value}
            for name in ("a", "b", "c")
            for value in (0, -1, 1000.1, np.inf, np.nan)
        ),
        *(
            {f"lattice_{name}": value}
            for name in ("alpha", "beta", "gamma")
            for value in (0, 180, -1, np.inf, np.nan)
        ),
        {"lattice_alpha": 10, "lattice_beta": 10, "lattice_gamma": 90},
        {"lattice_alpha": 30, "lattice_beta": 30, "lattice_gamma": 60},
        {"lattice_a": 1e-4, "lattice_b": 1e-4, "lattice_c": 1e-4},
        {"centering": "invalid"},
        {"lattice_rotation": np.nan},
        {"out_of_plane_range": ((0.0, 1.0), (2.0, 2.0))},
    ],
)
def test_three_dimensional_state_validation(fields):
    from erlab.interactive.angle_calculator._state import AngleCalculatorState

    with pytest.raises(ValueError, match="validation error"):
        AngleCalculatorState(**fields)


@pytest.mark.parametrize("energy_difference", [-0.7, 0.0, 0.7])
def test_position_limit_normal_inversion(energy_difference):
    from erlab.interactive.angle_calculator._low_energy import (
        slopes_to_momentum_at_normal,
    )

    sample_normal = np.array([-0.2, 0.1, 0.5, 0.8])[:, None]
    slope_x = np.array([-0.2, 0.0, 0.3])[None, :]
    slope_y = 0.15
    kx, ky = slopes_to_momentum_at_normal(
        slope_x, slope_y, sample_normal, energy_difference
    )
    assert np.isnan(kx[0]).all()
    assert np.isfinite(kx[2:]).all()
    if energy_difference < 0.0:
        assert np.isnan(kx[1]).all()
    sample_energy = (sample_normal**2 + kx**2 + ky**2) / erlab.constants.rel_kconv**2
    actual = momentum_to_slopes(
        kx, ky, sample_energy, sample_energy + energy_difference
    )
    valid = np.isfinite(kx)
    np.testing.assert_allclose(
        actual[0][valid], np.broadcast_to(slope_x, kx.shape)[valid]
    )
    np.testing.assert_allclose(actual[1][valid], slope_y)


@pytest.mark.parametrize("configuration", [1, 2])
def test_alpha_kz_surface_rejects_corrected_mechanical_scan(configuration):
    from erlab.interactive.angle_calculator._calculation import alpha_kz_surface

    geometry = Geometry(configuration=configuration, low_energy_correction=True)
    with pytest.raises(ValueError, match="zero mechanical beta"):
        alpha_kz_surface([0.0], [2.0], 1.0, geometry)


@pytest.mark.parametrize("corrected", [False, True])
def test_single_photon_energy_range(corrected):
    from erlab.interactive.angle_calculator._calculation import (
        alpha_kz_surface,
        angles_to_momentum3d,
    )
    from erlab.interactive.angle_calculator._state import AngleCalculatorState

    state = AngleCalculatorState(
        hv_range=(21.2, 21.2),
        geometry=Geometry(
            low_energy_correction=corrected,
            sample_work_function=5.2,
            analyzer_work_function=4.5,
            binding_energy=-0.3,
        ),
    )
    assert AngleCalculatorState.model_validate_json(state.model_dump_json()) == state
    hv = np.linspace(*state.hv_range, 5)
    forward = angles_to_momentum3d(5.0, 0.0, state.geometry, hv)
    assert np.isfinite(forward).all()
    surface, inferred_hv = alpha_kz_surface([5.0], [forward[2][0]], 0.0, state.geometry)
    np.testing.assert_allclose(surface[0, 0], np.asarray(forward)[:, 0], atol=1e-12)
    np.testing.assert_allclose(inferred_hv, state.hv_range[0])


def test_photon_range_minimum_state():
    from erlab.interactive.angle_calculator._state import AngleCalculatorState

    state = AngleCalculatorState(hv_range=(5, 5), geometry=Geometry(photon_energy=1))
    assert state.hv_range == (5, 5)
    assert state.geometry.photon_energy == 1
