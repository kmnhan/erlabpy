import pytest
from erlab.io.exampledata import generate_data_angles, generate_gold_edge


@pytest.fixture()
def anglemap():
    return generate_data_angles(shape=(10, 10, 10), assign_attributes=True)


@pytest.fixture()
def gold():
    return generate_gold_edge(
        temp=100, seed=1, nx=15, ny=150, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )
