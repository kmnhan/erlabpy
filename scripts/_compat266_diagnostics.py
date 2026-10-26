"""CI diagnostics for the native crash in compatibility run 266."""

import gc
import os

import pytest

_TARGET = (
    "tests/interactive/imagetool/manager/test_modelview.py::"
    "test_manager_open_files[dragdrop]"
)


def pytest_sessionstart(session):
    if os.environ["ERLAB_COMPAT266_GC_MODE"] == "frequent":
        gc.set_threshold(100, 5, 5)
    print(f"Compatibility 266 diagnostic GC thresholds: {gc.get_threshold()}")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    for index, item in enumerate(items):
        if item.nodeid == _TARGET:
            deselected = items[index + 1 :]
            del items[index + 1 :]
            config.hook.pytest_deselected(items=deselected)
            return
    raise pytest.UsageError(f"Diagnostic target was not collected: {_TARGET}")
