import importlib
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
