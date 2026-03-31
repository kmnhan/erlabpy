import importlib.util
import pathlib
import types


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
    def __init__(self, plugins: set[str]) -> None:
        self.pluginmanager = _DummyPluginManager(plugins)


def test_coverage_is_active_requires_internal_cov_plugin() -> None:
    assert not _CONFTEST._coverage_is_active(_DummyPytestConfig({"pytest_cov"}))
    assert _CONFTEST._coverage_is_active(_DummyPytestConfig({"pytest_cov", "_cov"}))
