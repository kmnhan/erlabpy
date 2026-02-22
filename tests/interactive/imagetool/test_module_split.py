import erlab
import erlab.interactive.imagetool.plot_items
import erlab.interactive.imagetool.viewer


def test_core_shim_export_identity() -> None:
    assert (
        erlab.interactive.imagetool.core.ImageSlicerArea
        is erlab.interactive.imagetool.viewer.ImageSlicerArea
    )
    assert (
        erlab.interactive.imagetool.core._parse_input
        is erlab.interactive.imagetool.viewer._parse_input
    )
    assert (
        erlab.interactive.imagetool.core._PolyROIEditDialog
        is erlab.interactive.imagetool.plot_items._PolyROIEditDialog
    )


def test_canonical_modules_are_importable() -> None:
    assert erlab.interactive.imagetool.viewer.ImageSlicerArea
    assert erlab.interactive.imagetool.plot_items.ItoolPlotItem
