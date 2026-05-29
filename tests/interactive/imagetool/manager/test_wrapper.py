# ruff: noqa: F403, F405
from ._shared import *


def test_wrapper_preview_fallback_branches(monkeypatch) -> None:
    fallback = QtGui.QPixmap(3, 2)
    fallback.fill(QtGui.QColor("red"))
    rendered = QtGui.QPixmap(4, 6)
    rendered.fill(QtGui.QColor("blue"))
    invalid = object()

    class _FakeImageItem:
        def __init__(
            self, pixmap: QtGui.QPixmap | None = None, *, raise_pixmap: bool = False
        ) -> None:
            self.pixmap = pixmap if pixmap is not None else rendered
            self.raise_pixmap = raise_pixmap

        def getPixmap(self) -> QtGui.QPixmap:
            if self.raise_pixmap:
                raise RuntimeError("pixmap unavailable")
            return self.pixmap

    class _FakeViewBox:
        def __init__(self, width: float, height: float) -> None:
            self._rect = QtCore.QRectF(0.0, 0.0, width, height)

        def rect(self) -> QtCore.QRectF:
            return self._rect

    class _FakeMainImage:
        def __init__(
            self,
            *,
            view_box: object = None,
            items: list[object] | None = None,
        ) -> None:
            self._view_box = (
                view_box if view_box is not None else _FakeViewBox(2.0, 8.0)
            )
            self.slicer_data_items = [_FakeImageItem()] if items is None else items

        def getViewBox(self) -> object:
            return self._view_box

    class _FakeSlicerArea:
        def __init__(
            self, main_image: object = None, *, raise_main: bool = False
        ) -> None:
            self._main_image = _FakeMainImage() if main_image is None else main_image
            self._raise_main = raise_main

        def _update_if_delayed(self) -> None:
            return

        @property
        def main_image(self) -> object:
            if self._raise_main:
                raise RuntimeError("main image unavailable")
            return self._main_image

    class _FakeImageTool:
        def __init__(self, slicer_area: _FakeSlicerArea) -> None:
            self.slicer_area = slicer_area

    def _preview(slicer_area: _FakeSlicerArea) -> tuple[float, QtGui.QPixmap]:
        return _preview_from_imagetool(
            typing.cast(
                "erlab.interactive.imagetool.ImageTool",
                _FakeImageTool(slicer_area),
            ),
            1.5,
            fallback,
        )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "qt_is_valid",
        lambda obj, *_args: obj is not invalid,
    )

    assert _preview_from_imagetool(None, 1.5, fallback) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(raise_main=True)) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(invalid)) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(_FakeMainImage(view_box=invalid))) == (
        1.5,
        fallback,
    )
    assert _preview(
        _FakeSlicerArea(_FakeMainImage(view_box=_FakeViewBox(0.0, 2.0)))
    ) == (
        1.5,
        fallback,
    )
    assert _preview(_FakeSlicerArea(_FakeMainImage(items=[]))) == (1.5, fallback)
    assert _preview(_FakeSlicerArea(_FakeMainImage(items=[invalid]))) == (1.5, fallback)
    assert _preview(
        _FakeSlicerArea(_FakeMainImage(items=[_FakeImageItem(raise_pixmap=True)]))
    ) == (1.5, fallback)
    assert _preview(
        _FakeSlicerArea(_FakeMainImage(items=[_FakeImageItem(QtGui.QPixmap())]))
    ) == (1.5, fallback)

    ratio, pixmap = _preview(_FakeSlicerArea(_FakeMainImage()))
    assert ratio == 4.0
    assert not pixmap.isNull()


def test_wrapper_loader_code_and_metadata_helper_branches(
    tmp_path: pathlib.Path,
) -> None:
    import math

    file_path = tmp_path / "data.nc"

    def _local_loader() -> None:
        return None

    def _missing_module_loader() -> None:
        return None

    _missing_module_loader.__module__ = "missing_erlab_loader_module"
    _missing_module_loader.__qualname__ = "load"

    def _missing_attr_loader() -> None:
        return None

    _missing_attr_loader.__module__ = "math"
    _missing_attr_loader.__qualname__ = "missing_loader"
    dataarray_selection = (
        erlab.interactive.imagetool.provenance_framework.FileDataSelection(
            kind="dataarray"
        )
    )

    assert _load_code_from_file_details(file_path, None) is None
    assert _load_provenance_from_file_details(file_path, None) is None
    provenance = _load_provenance_from_file_details(
        file_path,
        (xr.load_dataarray, {"engine": "h5netcdf"}, 0),
    )
    assert provenance is not None
    assert provenance.kind == "file"
    assert provenance.file_load_source is not None
    assert provenance.file_load_source.replay_call is not None
    assert provenance.file_load_source.replay_call.target == "xarray.load_dataarray"
    selected_code = _load_code_from_file_details(file_path, ("example", {}, 1))
    assert selected_code == (
        "erlab.io.set_loader('example')\n"
        "data = erlab.interactive.imagetool.viewer_state._parse_input("
        f"erlab.io.load({str(file_path)!r}))[1]"
    )
    assert _load_code_from_file_details(file_path, (_local_loader, {}, 0)) is None
    assert _load_provenance_from_file_details(file_path, (_local_loader, {}, 0)) is None
    assert _loader_callable_text(_local_loader) is None
    assert _load_source_label_and_text(None) == ("Loader", "(unavailable)")
    assert _load_source_label_and_text(("example", {}, 0)) == ("Loader", "example")
    assert _load_source_label_and_text((_local_loader, {}, 0)) == (
        "Load Function",
        repr(_local_loader),
    )
    assert _loader_callable_text(_missing_module_loader) == (
        "missing_erlab_loader_module.load"
    )
    assert _loader_callable_text(_missing_attr_loader) == "math.missing_loader"

    math_code = _load_code_from_file_details(
        file_path,
        (math.sqrt, {"bad-key": 1}, dataarray_selection),
    )
    assert math_code == (
        f'import math\n\ndata = math.sqrt({str(file_path)!r}, **{{"bad-key": 1}})'
    )

    missing_module_code = _load_code_from_file_details(
        file_path,
        (_missing_module_loader, {}, dataarray_selection),
    )
    assert missing_module_code == (
        "import missing_erlab_loader_module\n\n"
        f"data = missing_erlab_loader_module.load({str(file_path)!r})"
    )

    chunked = xr.DataArray(
        np.zeros((5, 4)),
        dims=("x", "y"),
    ).chunk({"x": (2, 3), "y": 2})
    assert _format_chunk_summary(xr.DataArray(np.zeros(2), dims=("x",))) == "In memory"
    assert _format_chunk_summary(chunked) == "x=2, 3; y=2"


def test_wrapper_source_data_replaced_uses_parent_fallback_and_skips_missing_child(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )

        wrapper = manager._imagetool_wrappers[0]
        _, child = next(iter(wrapper._childtools.items()))
        updated = test_data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 7
        handled: list[xr.DataArray] = []

        monkeypatch.setattr(
            wrapper.slicer_area, "_tool_source_parent_data", lambda: updated
        )
        monkeypatch.setattr(
            child, "handle_parent_source_replaced", lambda data: handled.append(data)
        )
        wrapper._childtool_indices.append("missing")

        wrapper._handle_source_data_replaced(object())

        assert handled == [updated]
