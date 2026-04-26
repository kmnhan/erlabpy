import importlib
import multiprocessing
import pathlib
import runpy
import sys
import types
import warnings

import erlab
import erlab.interactive.imagetool.plot_items
import erlab.interactive.imagetool.viewer


def test_core_shim_export_identity_and_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        core = importlib.import_module("erlab.interactive.imagetool.core")
        importlib.reload(core)

    assert any(issubclass(w.category, FutureWarning) for w in caught)

    assert core.ImageSlicerArea is erlab.interactive.imagetool.viewer.ImageSlicerArea
    assert core._parse_input is erlab.interactive.imagetool.viewer._parse_input
    assert (
        core._PolyROIEditDialog
        is erlab.interactive.imagetool.plot_items._PolyROIEditDialog
    )


def test_canonical_modules_are_importable() -> None:
    assert erlab.interactive.imagetool.viewer.ImageSlicerArea
    assert erlab.interactive.imagetool.plot_items.ItoolPlotItem


def test_manager_entrypoint_freeze_support_runs_before_manager_import(
    monkeypatch,
) -> None:
    events: list[str] = []

    manager_module = types.ModuleType("erlab.interactive.imagetool.manager")

    def _getattr(name: str):
        if name == "main":
            events.append("manager_import")
            return lambda: events.append("main")
        raise AttributeError(name)

    manager_module.__getattr__ = _getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "erlab.interactive.imagetool.manager", manager_module
    )
    monkeypatch.setattr(
        multiprocessing, "freeze_support", lambda: events.append("freeze")
    )

    entrypoint = (
        pathlib.Path(__file__).resolve().parents[3]
        / "src/erlab/interactive/imagetool/manager/__main__.py"
    )
    runpy.run_path(str(entrypoint), run_name="__main__")

    assert events == ["freeze", "manager_import", "main"]
