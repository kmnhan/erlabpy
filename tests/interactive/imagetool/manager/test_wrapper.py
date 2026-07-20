import datetime
import logging
import pathlib
import types
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui

import erlab
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._load_source import (
    _load_code_from_file_details,
    _load_provenance_from_file_details,
    _load_source_label_and_text,
    _loader_callable_text,
)
from erlab.interactive.imagetool._provenance._model import FileDataSelection
from erlab.interactive.imagetool.manager._wrapper import (
    _coerce_added_time,
    _coerce_note,
    _format_added_time,
    _format_chunk_summary,
    _ManagedWindowNode,
    _preview_from_imagetool,
    _preview_image_for_node,
)


def test_reapplying_workspace_link_state_keeps_color_cache_valid() -> None:
    invalidated: list[None] = []
    node = types.SimpleNamespace(
        _workspace_link_key="link-group",
        _workspace_link_colors=True,
        manager=types.SimpleNamespace(
            _invalidate_workspace_link_color_cache=lambda: invalidated.append(None)
        ),
    )

    _ManagedWindowNode.set_workspace_link_state(node, "link-group", link_colors=False)

    assert invalidated == []
    assert node._workspace_link_key == "link-group"
    assert node._workspace_link_colors is False


def test_wrapper_preview_fallback_branches(monkeypatch) -> None:
    fallback = QtGui.QPixmap(3, 2)
    fallback.fill(QtGui.QColor("red"))
    rendered = QtGui.QPixmap(4, 6)
    rendered.fill(QtGui.QColor("blue"))
    invalid = object()
    default_pixmap = object()

    class _FakeImageItem:
        def __init__(
            self,
            pixmap: QtGui.QPixmap | None | object = default_pixmap,
            *,
            raise_pixmap: bool = False,
        ) -> None:
            self.pixmap = (
                rendered
                if pixmap is default_pixmap
                else typing.cast("QtGui.QPixmap | None", pixmap)
            )
            self.raise_pixmap = raise_pixmap

        def getPixmap(self) -> QtGui.QPixmap | None:
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
    assert _preview(_FakeSlicerArea(_FakeMainImage(items=[_FakeImageItem(None)]))) == (
        1.5,
        fallback,
    )
    assert _preview(
        _FakeSlicerArea(_FakeMainImage(items=[_FakeImageItem(QtGui.QPixmap())]))
    ) == (1.5, fallback)

    ratio, pixmap = _preview(_FakeSlicerArea(_FakeMainImage()))
    assert ratio == 4.0
    assert not pixmap.isNull()


def test_wrapper_note_coercion_and_setter_validation(
    caplog,
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    assert _coerce_note(None) == ""
    assert _coerce_note(b"encoded note") == "encoded note"
    assert _coerce_note("plain note") == "plain note"

    with caplog.at_level(logging.WARNING):
        assert _coerce_note(b"\xff") == ""
        assert _coerce_note(1) == ""
    assert "Ignoring invalid saved manager note" in caplog.text

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        index = manager.add_imagetool(tool, show=False, note=b"stored note")
        wrapper = manager._tool_graph.root_wrappers[index]
        assert wrapper.note == "stored note"
        assert wrapper.has_note
        wrapper.note = "   "
        assert not wrapper.has_note
        with pytest.raises(TypeError, match="note must be a string"):
            wrapper.note = typing.cast("str", object())


def test_preview_image_for_node_handles_legacy_preview_nodes(monkeypatch) -> None:
    rendered = QtGui.QPixmap(4, 6)
    rendered.fill(QtGui.QColor("blue"))
    image_tool = object()
    calls: list[object | None] = []

    def _preview_from_imagetool_probe(
        imagetool,
        fallback_ratio: float,
        fallback_pixmap: QtGui.QPixmap,
    ) -> tuple[float, QtGui.QPixmap]:
        calls.append(imagetool)
        assert not np.isfinite(fallback_ratio)
        assert fallback_pixmap.isNull()
        return 2.0, rendered

    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._wrapper._preview_from_imagetool",
        _preview_from_imagetool_probe,
    )

    class _LegacyNode:
        @property
        def imagetool(self) -> object:
            return image_tool

    class _BrokenPreviewNode(_LegacyNode):
        @property
        def _preview_image(self) -> tuple[float, QtGui.QPixmap]:
            raise AttributeError("_preview_image")

    class _CachedPreviewNode:
        @property
        def _preview_image(self) -> tuple[float, QtGui.QPixmap]:
            return 3.0, rendered

    assert _preview_image_for_node(_LegacyNode()) == (2.0, rendered)
    assert _preview_image_for_node(_BrokenPreviewNode()) == (2.0, rendered)
    assert _preview_image_for_node(_CachedPreviewNode()) == (3.0, rendered)
    assert calls == [image_tool, image_tool]


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
    dataarray_selection = FileDataSelection(kind="dataarray")

    assert _load_code_from_file_details(file_path, None) is None
    assert _load_provenance_from_file_details(file_path, None) is None
    with pytest.raises(TypeError, match="FileDataSelection"):
        _load_code_from_file_details(
            file_path,
            typing.cast("typing.Any", ("example", {}, 0)),
        )
    provenance = _load_provenance_from_file_details(
        file_path,
        (
            xr.load_dataarray,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="dataarray"),
        ),
    )
    assert provenance is not None
    assert provenance.kind == "file"
    assert provenance.file_load_source is not None
    assert provenance.file_load_source.replay_call is not None
    assert provenance.file_load_source.replay_call.target == "xarray.load_dataarray"
    selected_code = _load_code_from_file_details(
        file_path,
        ("example", {}, FileDataSelection(kind="sequence_index", value=1)),
    )
    assert selected_code is not None
    assert "data_loaded" not in selected_code
    assert "isinstance" not in selected_code
    assert "imagetool" not in selected_code

    expected = xr.DataArray([3.0, 4.0], dims="x")
    selected_loaders: list[str] = []
    fake_io = types.SimpleNamespace(
        set_loader=selected_loaders.append,
        load=lambda _path: [xr.DataArray([1.0, 2.0], dims="x"), expected],
    )
    namespace = {"erlab": types.SimpleNamespace(io=fake_io)}
    exec(selected_code, namespace)  # noqa: S102
    assert selected_loaders == ["example"]
    xr.testing.assert_identical(namespace["data"], expected)
    assert (
        _load_code_from_file_details(
            file_path,
            ("example", {}, FileDataSelection(kind="parsed_index", value=0)),
        )
        is None
    )
    assert (
        _load_provenance_from_file_details(
            file_path,
            ("example", {}, FileDataSelection(kind="parsed_index", value=0)),
        )
        is None
    )
    assert (
        _load_code_from_file_details(
            file_path,
            (_local_loader, {}, dataarray_selection),
        )
        is None
    )
    assert (
        _load_provenance_from_file_details(
            file_path,
            (_local_loader, {}, dataarray_selection),
        )
        is None
    )
    assert _loader_callable_text(_local_loader) is None
    assert _load_source_label_and_text(None) == ("Loader", "(unavailable)")
    assert _load_source_label_and_text(("example", {}, dataarray_selection)) == (
        "Loader",
        "example",
    )
    assert _load_source_label_and_text((_local_loader, {}, dataarray_selection)) == (
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


def test_added_time_helpers_use_aware_datetimes(caplog) -> None:
    added = datetime.datetime(
        2024,
        1,
        2,
        3,
        4,
        5,
        987654,
        tzinfo=datetime.timezone(datetime.timedelta(hours=9), "KST"),
    )

    assert _coerce_added_time(added) == added.replace(microsecond=0)
    assert _coerce_added_time(added.isoformat().encode()) == added.replace(
        microsecond=0
    )
    assert _format_added_time(added) == added.astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z (%z)"
    )
    fallback = _coerce_added_time(None)
    assert fallback.tzinfo is not None
    assert fallback.utcoffset() is not None

    caplog.set_level(logging.WARNING)
    for invalid in (
        "not-a-date",
        b"\xff",
        object(),
        datetime.datetime(2024, 1, 2, 3, 4, 5),
    ):
        fallback = _coerce_added_time(invalid, node_uid="bad-node")
        assert fallback.tzinfo is not None
        assert fallback.utcoffset() is not None

    assert (
        "Ignoring invalid saved manager added timestamp for node bad-node"
        in caplog.text
    )


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
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        wrapper = manager._tool_graph.root_wrappers[0]
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

        assert len(handled) == 1
        xr.testing.assert_identical(handled[0], updated)
