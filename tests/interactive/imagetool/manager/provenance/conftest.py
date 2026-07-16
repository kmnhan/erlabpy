import pytest

from tests.interactive.imagetool.manager.helpers import (
    InMemoryClipboard,
    install_in_memory_clipboard,
)


@pytest.fixture(autouse=True)
def isolate_qt_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> InMemoryClipboard:
    return install_in_memory_clipboard(monkeypatch)
