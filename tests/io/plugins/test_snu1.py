import numpy as np
import pytest

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("snu1")
    erlab.io.set_data_dir(test_data_dir / "snu1")
    return test_data_dir / "snu1"


def test_load(data_dir) -> None:
    dat = erlab.io.load("data_001.itx")
    np.testing.assert_allclose(dat.alpha.values, np.linspace(-15, 15, len(dat.alpha)))
    np.testing.assert_allclose(dat.eV.values, np.linspace(16, 17, len(dat.eV)))

    dat = erlab.io.load("data_s.itx")
    np.testing.assert_allclose(dat.Step.values, np.linspace(1, 8, len(dat.Step)))
    np.testing.assert_allclose(dat.eV.values, np.linspace(16, 17, len(dat.eV)))
