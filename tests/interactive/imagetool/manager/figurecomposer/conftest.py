import matplotlib.pyplot as plt
import pytest

from erlab.interactive._options import options
from erlab.interactive._options.schema import AppOptions
from tests.interactive.imagetool.manager.helpers import (
    InMemoryClipboard,
    install_in_memory_clipboard,
)


@pytest.fixture(autouse=True)
def restore_interactive_options():
    old_options = options.model
    options.model = AppOptions()
    try:
        yield
    finally:
        options.model = old_options
        plt.close("all")


@pytest.fixture(autouse=True)
def isolate_qt_clipboard(monkeypatch: pytest.MonkeyPatch) -> InMemoryClipboard:
    return install_in_memory_clipboard(monkeypatch)
