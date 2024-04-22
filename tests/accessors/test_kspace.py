import erlab.analysis.kspace
import pytest
import xarray
from erlab.io.exampledata import generate_data_angles


@pytest.fixture()
def angle_data():
    return generate_data_angles(shape=(10, 10, 10), assign_attributes=True)


def test_offsets(angle_data):
    angle_data.kspace.offsets.reset()
    angle_data.kspace.offsets = {"xi": 10.0}
    answer = dict.fromkeys(angle_data.kspace.valid_offset_keys, 0.0)
    answer["xi"] = 10.0
    assert dict(angle_data.kspace.offsets) == answer


def test_kconv(angle_data):
    data = angle_data.copy()

    for conf in erlab.analysis.kspace.AxesConfiguration:
        data.kspace.configuration = conf
        match conf:
            case (
                erlab.analysis.kspace.AxesConfiguration.Type1DA
                | erlab.analysis.kspace.AxesConfiguration.Type2DA
            ):
                data = data.assign_coords(chi=0.0)

        kconv = data.kspace.convert()

        assert isinstance(kconv, xarray.DataArray)
        assert not kconv.isnull().all()


def test_kconv_kinetic(angle_data):
    data = angle_data.copy()
    data = data.assign_coords(eV=data.hv - data.kspace.work_function + data.eV)

    for conf in erlab.analysis.kspace.AxesConfiguration:
        data.kspace.configuration = conf
        match conf:
            case (
                erlab.analysis.kspace.AxesConfiguration.Type1DA
                | erlab.analysis.kspace.AxesConfiguration.Type2DA
            ):
                data = data.assign_coords(chi=0.0)

        kconv = data.kspace.convert()

        assert isinstance(kconv, xarray.DataArray)
        assert not kconv.isnull().all()
