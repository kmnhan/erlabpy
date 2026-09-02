import matplotlib.pyplot as plt
import pytest

from erlab.interactive._options import options
from erlab.interactive._options.schema import AppOptions


@pytest.fixture(autouse=True)
def restore_interactive_options():
    old_options = options.model
    options.model = AppOptions()
    try:
        yield
    finally:
        options.model = old_options
        plt.close("all")
