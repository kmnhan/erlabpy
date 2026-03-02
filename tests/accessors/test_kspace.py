import re

import numpy as np
import pytest
import xarray
import xarray.testing

from erlab.accessors.kspace import IncompleteDataError
from erlab.constants import AxesConfiguration
from erlab.io.exampledata import generate_hvdep_cuts


@pytest.fixture(scope="module")
def hvdep():
    data = generate_hvdep_cuts((50, 250, 300), seed=1)
    data.kspace.inner_potential = 10.0
    return data


@pytest.fixture(scope="module")
def config_1_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 1
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_2_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 2
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_3_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_4_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.attrs["configuration"] = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_1_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 1
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_2_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 2
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_3_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_4_cut(cut):
    data = cut.copy(deep=True)
    data.attrs["configuration"] = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_1_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 1
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_2_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 2
    return data.kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_3_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture(scope="module")
def config_4_hvdep(hvdep):
    data = hvdep.copy(deep=True)
    data.attrs["configuration"] = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.mark.parametrize("data_type", ["anglemap", "cut", "hvdep"])
def test_offsets(data_type, request) -> None:
    data = request.getfixturevalue(data_type).copy(deep=True)
    data.kspace.offsets.reset()
    data.kspace.offsets = {"xi": 10.0}
    answer = dict.fromkeys(data.kspace._valid_offset_keys, 0.0)
    answer["xi"] = 10.0
    assert dict(data.kspace.offsets) == answer

    with pytest.raises(
        KeyError,
        match=re.escape(
            "Invalid offset key 'invalid' for experimental configuration "
            f"{data.kspace.configuration}. Valid keys are: "
            f"{data.kspace._valid_offset_keys}."
        ),
    ):
        data.kspace.offsets["invalid"] = 10.0


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no-dask"])
@pytest.mark.parametrize("energy_axis", ["kinetic", "binding"])
@pytest.mark.parametrize("data_type", ["anglemap", "cut", "hvdep"])
@pytest.mark.parametrize(
    "configuration", AxesConfiguration, ids=[c.name for c in AxesConfiguration]
)
@pytest.mark.parametrize("extra_dims", [0, 1, 2], ids=["extra0", "extra1", "extra2"])
def test_kconv(
    use_dask: bool,
    energy_axis: str,
    data_type: str,
    configuration: AxesConfiguration,
    extra_dims: int,
    request,
) -> None:
    data = request.getfixturevalue(data_type).copy(deep=True)

    expected = request.getfixturevalue(
        f"config_{configuration.value}_{data_type.replace('angle', 'k')}"
    )

    if energy_axis == "kinetic":
        data = data.assign_coords(eV=data.hv - data.kspace.work_function + data.eV)

    if extra_dims > 0:
        data = data.expand_dims({f"extra{i}": 2 for i in range(extra_dims)})

    if use_dask:
        data = data.chunk()

    data.attrs["configuration"] = configuration.value

    for c in AxesConfiguration:
        data_ = data.kspace.as_configuration(c)
        if c == configuration:
            xarray.testing.assert_identical(data_, data)
        else:
            assert data_.attrs["configuration"] == c.value

    del data_

    match configuration:
        case AxesConfiguration.Type1DA | AxesConfiguration.Type2DA:
            data = data.assign_coords(chi=0.0)

    if data_type == "hvdep":
        data.kspace.inner_potential = 10.0

    if energy_axis == "kinetic":
        if data_type == "hvdep":
            with pytest.raises(
                ValueError,
                match=r"Energy axis of photon energy dependent data must be in "
                r"binding energy.",
            ):
                kconv = data.kspace.convert(silent=False)
            return

        with pytest.warns(
            UserWarning,
            match="The energy axis seems to be in terms of kinetic energy, "
            "attempting conversion to binding energy.",
        ):
            kconv = data.kspace.convert(silent=False)
    else:
        kconv = data.kspace.convert(silent=False)

    if use_dask:
        kconv = kconv.compute()

    if extra_dims > 0:
        for j in range(1, extra_dims):
            xarray.testing.assert_allclose(
                expected, kconv.isel({f"extra{i}": j for i in range(extra_dims)})
            )
    else:
        xarray.testing.assert_allclose(expected, kconv)

    if data_type == "anglemap":
        assert len(kconv.shape) == 3 + extra_dims
        if extra_dims == 0:
            assert set(kconv.shape) == {10, 310}
        else:
            assert set(kconv.shape) == {10, 310, 2}

    elif data_type == "cut":
        assert len(kconv.shape) == 2 + extra_dims
        if extra_dims == 0:
            assert set(kconv.shape) == {310, 500}
        else:
            assert set(kconv.shape) == {310, 500, 2}

    elif data_type == "hvdep":
        assert len(kconv.shape) == 3 + extra_dims
        if extra_dims == 0:
            assert set(kconv.shape) == {637, 63, 300}
        else:
            assert set(kconv.shape) == {637, 63, 300, 2}

    assert isinstance(kconv, xarray.DataArray)
    assert not kconv.isnull().all()


@pytest.mark.parametrize("missing_coord", ["alpha", "beta", "chi", "xi", "eV", "hv"])
def test_kconv_missing_coord(missing_coord, anglemap):
    data = anglemap.copy().assign_coords(chi=0.0)
    data.attrs["configuration"] = 4

    data = data.drop_vars(missing_coord)

    with pytest.raises(
        IncompleteDataError,
        match=IncompleteDataError._make_message("coord", missing_coord),
    ):
        data.kspace.convert(silent=True)


@pytest.mark.parametrize("missing_attr", ["configuration"])
def test_kconv_missing_attr(missing_attr, anglemap):
    data = anglemap.copy().assign_coords(chi=0.0)
    data.attrs["configuration"] = 4
    data.attrs.pop(missing_attr)

    with pytest.raises(
        IncompleteDataError,
        match=IncompleteDataError._make_message("attr", missing_attr),
    ):
        data.kspace.convert(silent=True)


def test_kspace_set_existing_configuration(anglemap):
    data = anglemap.copy().assign_coords(chi=0.0)

    with pytest.raises(
        AttributeError,
        match=r"Configuration is already set. To modify the experimental "
        r"configuration, use `DataArray.kspace.as_configuration`.",
    ):
        data.kspace.configuration = 4


def test_resolution_raises_nonpositive_kinetic_energy(anglemap) -> None:
    data = anglemap.copy(deep=True)
    data.kspace.work_function = 49.7

    with pytest.raises(
        ValueError,
        match=r"Nonphysical kinetic energy detected while estimating in-plane "
        r"momentum resolution: min\(E_k\)=",
    ):
        _ = data.kspace.best_kp_resolution

    with pytest.raises(
        ValueError,
        match=r"Nonphysical kinetic energy detected while estimating out-of-plane "
        r"momentum resolution: min\(E_k\)=",
    ):
        _ = data.kspace.best_kz_resolution

    with pytest.raises(
        ValueError,
        match=r"Nonphysical kinetic energy detected while converting to momentum "
        r"space: min\(E_k\)=",
    ):
        _ = data.kspace.convert(silent=True)


def test_estimate_resolution_from_numpoints_kz_uses_adjacent_spacing(hvdep) -> None:
    data = hvdep.copy(deep=True)
    lims = data.kspace.estimate_bounds()["kz"]

    out = data.kspace.estimate_resolution("kz", lims=lims, from_numpoints=True)
    expected = float((lims[1] - lims[0]) / (data.hv.size - 1))

    assert out == pytest.approx(expected)


def test_estimate_resolution_from_numpoints_single_point_returns_inf(anglemap) -> None:
    data = anglemap.isel(alpha=slice(0, 1)).copy(deep=True)
    axis = data.kspace.slit_axis

    out = data.kspace.estimate_resolution(axis, lims=(0.0, 1.0), from_numpoints=True)

    assert np.isinf(out)


def test_finite_minmax_all_nonfinite(anglemap) -> None:
    data = anglemap.copy(deep=True)
    mn, mx = data.kspace._finite_minmax(np.array([np.nan, np.inf, -np.inf]))
    assert np.isnan(mn)
    assert np.isnan(mx)


def test_check_kinetic_energy_warns_without_raising_for_nonfinite(anglemap) -> None:
    data = anglemap.copy(deep=True)
    kinetic = np.array([np.nan, np.inf, -np.inf])
    with pytest.warns(
        UserWarning,
        match=r"Cannot proceed while test context: kinetic energy contains no finite "
        r"values.",
    ):
        result = data.kspace._check_kinetic_energy(
            context="test context",
            kinetic_energy=kinetic,
            raise_on_violation=False,
        )
    assert result is kinetic


def test_check_kinetic_energy_raises_for_nonfinite(anglemap) -> None:
    data = anglemap.copy(deep=True)
    with pytest.raises(
        ValueError,
        match=r"Cannot proceed while test context: kinetic energy contains no finite "
        r"values.",
    ):
        _ = data.kspace._check_kinetic_energy(
            context="test context",
            kinetic_energy=np.array([np.nan, np.inf, -np.inf]),
        )


def test_check_kinetic_energy_warns_without_raising_for_nonphysical(anglemap) -> None:
    data = anglemap.copy(deep=True)
    data.kspace.work_function = 49.7
    with pytest.warns(
        UserWarning,
        match=r"Nonphysical kinetic energy detected while test context: min\(E_k\)=",
    ):
        result = data.kspace._check_kinetic_energy(
            context="test context",
            raise_on_violation=False,
        )
    xarray.testing.assert_identical(result, data.kspace._kinetic_energy)
