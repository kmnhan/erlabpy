import importlib.util
import os
import pathlib
import subprocess
import sys
import types

import pytest


def _load_conftest_module() -> types.ModuleType:
    path = pathlib.Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("tests_conftest_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CONFTEST = _load_conftest_module()


class _DummyPluginManager:
    def __init__(self, plugins: set[str]) -> None:
        self._plugins = plugins

    def hasplugin(self, name: str) -> bool:
        return name in self._plugins


class _DummyPytestConfig:
    def __init__(self, plugins: set[str], cov_source: list[str] | None = None) -> None:
        self.pluginmanager = _DummyPluginManager(plugins)
        self.option = types.SimpleNamespace(
            cov_source=[] if cov_source is None else cov_source
        )


def test_coverage_is_active_requires_requested_cov_source() -> None:
    assert not _CONFTEST._coverage_is_active(_DummyPytestConfig({"pytest_cov"}))
    assert not _CONFTEST._coverage_is_active(_DummyPytestConfig({"pytest_cov", "_cov"}))
    assert _CONFTEST._coverage_is_active(
        _DummyPytestConfig({"pytest_cov", "_cov"}, ["erlab"])
    )


def test_conftest_import_defaults_pyside6_to_offscreen() -> None:
    if importlib.util.find_spec("PySide6") is None:
        pytest.skip("PySide6 is not installed")

    path = pathlib.Path(__file__).with_name("conftest.py")
    env = os.environ.copy()
    env.pop("QT_QPA_PLATFORM", None)
    env["DISPLAY"] = ":99.0"
    env["PYTEST_QT_API"] = "pyside6"
    env["QT_API"] = "pyside6"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, os, pathlib, sys; "
                "path = pathlib.Path(sys.argv[1]); "
                "spec = importlib.util.spec_from_file_location("
                "'platform_conftest', path); "
                "module = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module); "
                "print(os.environ['QT_QPA_PLATFORM'])"
            ),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip().splitlines()[-1] == "offscreen"


def test_is_deleted_qt_wrapper_error_matches_deleted_wrapper_message() -> None:
    exc = RuntimeError("wrapped C/C++ object of type InfiniteLine has been deleted")
    assert _CONFTEST._is_deleted_qt_wrapper_error(exc)
    assert not _CONFTEST._is_deleted_qt_wrapper_error(RuntimeError("different error"))


def test_collection_marker_hook_runs_before_xdist_loadgroup() -> None:
    hook_options = _CONFTEST.pytest_collection_modifyitems.pytest_impl
    assert hook_options["tryfirst"]


def test_serial_xdist_group_serializes_manager_context_tests() -> None:
    slicer_group = _CONFTEST.serial_xdist_group(
        "tests/interactive/imagetool/test_slicer.py",
        "tests/interactive/imagetool/test_slicer.py::test_array_rect",
    )
    workspace_group = _CONFTEST.serial_xdist_group(
        "tests/interactive/imagetool/manager/test_workspace.py",
        "tests/interactive/imagetool/manager/test_workspace.py::test_roundtrip",
    )
    explorer_group = _CONFTEST.serial_xdist_group(
        "tests/interactive/test_explorer.py",
        "tests/interactive/test_explorer.py::test_explorer_general",
    )
    watcher_group = _CONFTEST.serial_xdist_group(
        "tests/interactive/imagetool/test_watcher.py",
        "tests/interactive/imagetool/test_watcher.py::test_watcher_real",
    )
    console_group = _CONFTEST.serial_xdist_group(
        "tests/interactive/imagetool/manager/test_console.py",
        "tests/interactive/imagetool/manager/test_console.py::test_console",
    )

    assert slicer_group == "qt-tests-interactive-imagetool-test_slicer"
    assert workspace_group == "qt-tests-interactive-imagetool-manager-test_workspace"
    assert explorer_group == "qt-tests-interactive-test_explorer"
    assert watcher_group == "qt-tests-interactive-imagetool-test_watcher"
    assert console_group == "qt-tests-interactive-imagetool-manager-test_console"
    assert slicer_group != workspace_group
    assert console_group != workspace_group


def test_pyqtgraph_boundingrect_ignores_deleted_infinite_line(qtbot) -> None:
    import pyqtgraph as pg
    from qtpy import QtCore, QtWidgets

    from erlab.interactive.utils import qt_is_valid

    line = pg.InfiniteLine()
    line.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qtbot.waitUntil(lambda: not qt_is_valid(line), timeout=1000)

    assert line.boundingRect() == QtCore.QRectF()
