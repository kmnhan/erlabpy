import pytest
import xarray
import xarray.testing
from erlab.analysis.kspace import AxesConfiguration
from erlab.io.exampledata import generate_data_angles


@pytest.fixture()
def cut():
    return generate_data_angles(
        (300, 1, 500),
        angrange={"alpha": (-15, 15), "beta": (4.5, 4.5)},
        assign_attributes=True,
    ).T


@pytest.fixture()
def config_1_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.kspace.configuration = 1
    return data.kspace.convert(silent=True)


@pytest.fixture()
def config_2_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.kspace.configuration = 2
    return data.kspace.convert(silent=True)


@pytest.fixture()
def config_3_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.kspace.configuration = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture()
def config_4_kmap(anglemap):
    data = anglemap.copy(deep=True)
    data.kspace.configuration = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture()
def config_1_cut(cut):
    data = cut.copy(deep=True)
    data.kspace.configuration = 1
    return data.kspace.convert(silent=True)


@pytest.fixture()
def config_2_cut(cut):
    data = cut.copy(deep=True)
    data.kspace.configuration = 2
    return data.kspace.convert(silent=True)


@pytest.fixture()
def config_3_cut(cut):
    data = cut.copy(deep=True)
    data.kspace.configuration = 3
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.fixture()
def config_4_cut(cut):
    data = cut.copy(deep=True)
    data.kspace.configuration = 4
    return data.assign_coords(chi=0.0).kspace.convert(silent=True)


@pytest.mark.parametrize("data_type", ["anglemap", "cut"])
def test_offsets(data_type, request):
    data = request.getfixturevalue(data_type).copy(deep=True)
    data.kspace.offsets.reset()
    data.kspace.offsets = {"xi": 10.0}
    answer = dict.fromkeys(data.kspace.valid_offset_keys, 0.0)
    answer["xi"] = 10.0
    assert dict(data.kspace.offsets) == answer


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("energy_axis", ["kinetic", "binding"])
@pytest.mark.parametrize("data_type", ["anglemap", "cut"])
@pytest.mark.parametrize("configuration", AxesConfiguration)
@pytest.mark.parametrize("extra_dims", [0, 1, 2])
def test_kconv(
    use_dask: bool,
    energy_axis: str,
    data_type: str,
    configuration: AxesConfiguration,
    extra_dims: int,
    request,
):
    data = request.getfixturevalue(data_type).copy(deep=True)

    expected = request.getfixturevalue(
        f"config_{configuration.value}_{data_type.replace('angle', 'k')}"
    )

    if energy_axis == "kinetic":
        data = data.assign_coords(eV=data.hv - data.kspace.work_function + data.eV)

    if extra_dims > 0:
        data = data.expand_dims({f"extra{i}": 2 for i in range(extra_dims)})

    if use_dask:
        data = data.chunk("auto")

    data.kspace.configuration = configuration
    match configuration:
        case AxesConfiguration.Type1DA | AxesConfiguration.Type2DA:
            data = data.assign_coords(chi=0.0)

    kconv = data.kspace.convert(silent=True)

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

    else:
        assert len(kconv.shape) == 2 + extra_dims
        if extra_dims == 0:
            assert set(kconv.shape) == {310, 500}
        else:
            assert set(kconv.shape) == {310, 500, 2}

    assert isinstance(kconv, xarray.DataArray)
    assert not kconv.isnull().all()
