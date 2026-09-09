import json
import logging
import pathlib

import numpy as np
import pydantic
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive._qt_state as qt_state
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
from erlab.extensions._models import _script_name_key
from erlab.interactive import _persistence_constants
from erlab.interactive.imagetool._provenance._model import ScriptInput, script
from tests.interactive.imagetool.manager.workspace._support import (
    _workspace_test_file_spec,
)


def test_tool_data_blob_ignores_stale_backend_encoding() -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
        name="secondary",
    )
    data.encoding["compression"] = "unknown"
    data.encoding["source"] = "stale-source.nc"
    data.coords["x"].encoding["compression"] = "unknown"

    blob = erlab.interactive.utils._tool_data_to_blob(data, "secondary")
    restored = erlab.interactive.utils._tool_data_from_blob(blob)

    xr.testing.assert_equal(restored, data)
    assert data.encoding["compression"] == "unknown"
    assert data.coords["x"].encoding["compression"] == "unknown"


def test_tool_data_blob_preserves_none_name() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))

    blob = erlab.interactive.utils._tool_data_to_blob(data, "secondary")
    restored = erlab.interactive.utils._tool_data_from_blob(blob)

    assert restored.name is None


@pytest.mark.parametrize("name", [None, "source"])
def test_tool_data_blob_preserves_nested_attrs(name) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": [0.0, 1.0],
            "y": [0.0, 1.0, 2.0],
            "Fake Motor": ("x", [10.0, 20.0]),
        },
        name=name,
        attrs={"nested": {"values": [1, 2, 3]}},
    )
    data.coords["Fake Motor"].attrs["calibration"] = {
        "offset": None,
        "limits": (1.0, 2.0),
    }
    before = data.copy(deep=True)

    for _ in range(2):
        blob = erlab.interactive.utils._tool_data_to_blob(data, "secondary")
        restored = erlab.interactive.utils._tool_data_from_blob(blob)
        xr.testing.assert_identical(restored, before)
        xr.testing.assert_identical(data, before)
        data = restored


def _assert_tool_attr_equal(actual, expected) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_tool_attr_equal(actual[key], value)
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tool_attr_equal(left, right)
    elif isinstance(expected, np.str_ | np.bytes_):
        assert actual.dtype == expected.dtype
        assert len(actual) == len(expected)
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        if expected.dtype.kind == "O":
            for left, right in zip(actual.flat, expected.flat, strict=True):
                _assert_tool_attr_equal(left, right)
        else:
            np.testing.assert_array_equal(actual, expected, strict=True)
    else:
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"bytes": b"plain bytes"},
        b"\x00\xff",
        "text\x00tail",
        np.str_("sample\x00"),
        np.bytes_(b"sample\x00"),
        {
            "sample": 1,
            np.str_("sample\x00"): 2,
            np.str_(""): 3,
            np.str_("\x00"): 4,
            (np.str_("β\x00"), np.bytes_(b"\xff\x00")): 5,
        },
        {
            b"sample": 1,
            np.bytes_(b"sample\x00"): 2,
            np.bytes_(b""): 3,
            np.bytes_(b"\x00"): 4,
        },
        {
            "unicode": [
                np.str_(text)
                for text in (
                    "",
                    "plain",
                    "\x00",
                    "\x00\x00",
                    "a\x00b\x00\x00",
                    "\ufeffβ😀\ud800\x00",
                )
            ],
            "bytes": [
                np.bytes_(data)
                for data in (b"", b"plain", b"\x00", b"\x00\x00", b"a\x00b\xff\x00\x00")
            ],
            "empty_unicode_array": np.ndarray((2,), dtype="U0"),
            "empty_bytes_array": np.ndarray((0,), dtype="S0"),
        },
        [1, "text", None],
        ("text", b"bytes"),
        np.datetime64("2025-01-01", "ns"),
        np.timedelta64(3, "ns"),
        [np.timedelta64(1, "ns"), np.timedelta64(2, "ns")],
        np.array([1, 2], dtype="timedelta64[ns]"),
        np.array(["alpha", "β"]),
        {"array": np.array([b"a\x00b", b"\xff"], dtype="S3")},
        np.array(["a\x00b"]),
        np.empty(0, dtype=object),
        np.array([{"nested": (None, np.int16(3))}], dtype=object),
        {
            "array": np.arange(4, dtype=np.int16).reshape(2, 2),
            "scalar": np.float32(1.5),
            "tuple": (None, b"bytes", complex(2, 3)),
            "mapping": {(1, "x"): [False, float("inf"), float("nan")]},
        },
    ],
)
def test_tool_dataset_attrs_roundtrip_typed_values(value) -> None:
    ds = xr.Dataset(
        {"data": ("x", [1.0, 2.0])},
        coords={"x": [0, 1]},
        attrs={"metadata": value},
    )
    ds["data"].attrs["metadata"] = value
    ds["x"].attrs["metadata"] = value
    # Both transport keys are user metadata here. Even a plausible envelope is literal.
    ds.attrs[_persistence_constants.TOOL_ATTRS_VERSION_ATTR] = 99
    ds.attrs[_persistence_constants.TOOL_ENCODED_ATTRS_ATTR] = json.dumps(
        {"dataset": [], "variables": {}}
    )
    # NumPy's deep copy also strips trailing NULs from string scalars.
    original = ds.copy(deep=False)

    for _ in range(2):
        prepared = imagetool_serialization.prepare_tool_dataset(ds)
        raw = prepared.to_netcdf(engine="h5netcdf", invalid_netcdf=True)
        opened = xr.load_dataset(memoryview(raw), engine="h5netcdf")
        restored = imagetool_serialization.restore_tool_dataset_attrs(opened)
        _assert_tool_attr_equal(restored.attrs, original.attrs)
        _assert_tool_attr_equal(ds.attrs, original.attrs)
        for name in original.variables:
            _assert_tool_attr_equal(restored[name].attrs, original[name].attrs)
            _assert_tool_attr_equal(ds[name].attrs, original[name].attrs)
            np.testing.assert_array_equal(restored[name].values, original[name].values)
        ds = restored


def test_tool_dataset_native_attrs_keep_legacy_storage(monkeypatch) -> None:
    ds = xr.Dataset({"data": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    ds.attrs.update(
        title="source",
        count=3,
        valid=True,
        scalar=np.float32(1.5),
        numbers=[1, 2, 3],
        names=("one", "two"),
        empty=[],
        coefficients=np.array([1 + 2j, 3 + 4j]),
        bytes=b"text",
        bytes_array=np.array([b"one", b"two"]),
        object_strings=np.array(["one", "two"], dtype=object),
        object_bytes=np.array([b"one", b"two"], dtype=object),
        compound=np.array([(1, 2), (3, 4)], dtype=[("first", "i4"), ("second", "i4")]),
        void_array=np.array([b"ab", b"cd"], dtype="V2"),
        dtype_metadata=np.array([1, 2], dtype=np.dtype("i4", metadata={"units": "x"})),
    )
    ds["data"].attrs["units"] = "counts"

    def fail_if_encoded(value):
        raise AssertionError("Native attributes must not use the typed codec")

    monkeypatch.setattr(imagetool_serialization, "encode_attr_value", fail_if_encoded)
    prepared = imagetool_serialization.prepare_tool_dataset(ds)
    assert _persistence_constants.TOOL_ATTRS_VERSION_ATTR not in prepared.attrs
    assert imagetool_serialization.restore_tool_dataset_attrs(prepared) is prepared
    baseline = xr.load_dataset(
        memoryview(ds.to_netcdf(engine="h5netcdf", invalid_netcdf=True)),
        engine="h5netcdf",
    )
    restored = xr.load_dataset(
        memoryview(prepared.to_netcdf(engine="h5netcdf", invalid_netcdf=True)),
        engine="h5netcdf",
    )
    xr.testing.assert_identical(restored, baseline)


@pytest.mark.parametrize("lazy", [False, True])
def test_tool_dataset_metadata_preparation_shares_buffers(lazy) -> None:
    import dask.array
    from dask.callbacks import Callback

    data = np.arange(6.0).reshape(2, 3)
    coord = np.arange(6.0).reshape(2, 3) + 10
    if lazy:
        data = dask.array.from_array(data, chunks=(1, 3))
        coord = dask.array.from_array(coord, chunks=(1, 3))
    ds = xr.Dataset({"data": (("x", "y"), data)}, coords={"aux": (("x", "y"), coord)})
    ds["data"].attrs["metadata"] = {"values": [1, 2, 3]}
    ds["data"].encoding = {"compression": "unknown", "dtype": np.dtype("float64")}
    ds["aux"].encoding = {"source": "old-file"}
    attrs = ds["data"].attrs.copy()
    executed = []
    with Callback(posttask=lambda *args: executed.append(args)):
        prepared = imagetool_serialization.prepare_tool_dataset(
            ds, strip_backend_encoding=True
        )
        restored = imagetool_serialization.restore_tool_dataset_attrs(prepared)

    assert not executed
    for name in ds.variables:
        assert prepared[name].data is ds[name].data
        assert restored[name].data is ds[name].data
    assert prepared["data"].encoding == {"dtype": np.dtype("float64")}
    assert prepared["aux"].encoding == {}
    assert ds["data"].encoding["compression"] == "unknown"
    assert ds["aux"].encoding == {"source": "old-file"}
    assert ds["data"].attrs == attrs
    assert restored["data"].attrs == attrs


@pytest.mark.parametrize(
    "value",
    [
        object(),
        np.array([(1,)], dtype=[("field", "i4")]),
        np.array([(object(),)], dtype=[("field", object)]),
        np.array([1], dtype=np.dtype("i4", metadata={"units": "x"})),
        np.array(["short", "long" * 20], dtype=np.dtypes.StringDType()),
    ],
)
def test_tool_dataset_attrs_reject_unsupported_nested_values(value) -> None:
    ds = xr.Dataset({"data": ("x", [1.0])})
    metadata = {"nested": value}
    ds["data"].attrs["metadata"] = metadata
    with pytest.raises(TypeError, match="Cannot serialize tool attribute 'metadata'"):
        imagetool_serialization.prepare_tool_dataset(ds)
    assert ds["data"].attrs["metadata"] is metadata


@pytest.mark.parametrize(
    "value", [object(), np.array(["text"], dtype=np.dtypes.StringDType())]
)
def test_tool_dataset_attrs_reject_unsupported_direct_values(value) -> None:
    ds = xr.Dataset(attrs={"metadata": value})
    with pytest.raises(TypeError, match="Cannot serialize tool attribute 'metadata'"):
        imagetool_serialization.prepare_tool_dataset(ds)


@pytest.mark.parametrize("container", [list, tuple, lambda value: [value]])
def test_tool_dataset_native_structured_sequences(container) -> None:
    values = np.array([(1, 2.0), (3, 4.0)], dtype=[("count", "i4"), ("value", "f8")])
    ds = xr.Dataset(attrs={"metadata": container(list(values))})
    prepared = imagetool_serialization.prepare_tool_dataset(ds)
    assert _persistence_constants.TOOL_ATTRS_VERSION_ATTR not in prepared.attrs
    baseline = xr.load_dataset(
        memoryview(ds.to_netcdf(engine="h5netcdf", invalid_netcdf=True)),
        engine="h5netcdf",
    )
    restored = xr.load_dataset(
        memoryview(prepared.to_netcdf(engine="h5netcdf", invalid_netcdf=True)),
        engine="h5netcdf",
    )
    xr.testing.assert_identical(restored, baseline)


def test_tool_dataset_native_vlen_attributes() -> None:
    import h5py

    ds = xr.Dataset(
        attrs={"metadata": np.array(["first", "second"], dtype=h5py.string_dtype())}
    )
    prepared = imagetool_serialization.prepare_tool_dataset(ds)
    assert _persistence_constants.TOOL_ATTRS_VERSION_ATTR not in prepared.attrs
    baseline = xr.load_dataset(
        memoryview(ds.to_netcdf(engine="h5netcdf", invalid_netcdf=True)),
        engine="h5netcdf",
    )
    restored = xr.load_dataset(
        memoryview(prepared.to_netcdf(engine="h5netcdf", invalid_netcdf=True)),
        engine="h5netcdf",
    )
    xr.testing.assert_identical(restored, baseline)


def test_tool_dataset_attrs_reject_object_sequences_and_cycles() -> None:
    sequence = np.empty(1, dtype=object)
    sequence[0] = [1, 2]
    cycle = {}
    cycle["self"] = cycle
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    for value in (sequence, cycle, cyclic_list):
        ds = xr.Dataset(attrs={"metadata": value})
        with pytest.raises(
            TypeError, match="Cannot serialize tool attribute 'metadata'"
        ):
            imagetool_serialization.prepare_tool_dataset(ds)


@pytest.mark.parametrize("name", ["", 1])
def test_tool_dataset_attrs_reject_invalid_names(name) -> None:
    with pytest.raises(TypeError, match="non-empty strings"):
        imagetool_serialization.prepare_tool_dataset(xr.Dataset(attrs={name: "value"}))


def test_tool_dataset_attrs_reject_non_string_variable_name() -> None:
    ds = xr.Dataset({1: ("x", [1.0])})
    ds[1].attrs["metadata"] = None
    with pytest.raises(TypeError, match="variable names must be strings"):
        imagetool_serialization.prepare_tool_dataset(ds)


@pytest.mark.parametrize("version", [0, 2, True, "1", [1]])
def test_tool_dataset_attrs_reject_unknown_version(version) -> None:
    ds = xr.Dataset(attrs={_persistence_constants.TOOL_ATTRS_VERSION_ATTR: version})
    with pytest.raises(ValueError, match="Unsupported tool attribute encoding version"):
        imagetool_serialization.restore_tool_dataset_attrs(ds)


@pytest.mark.parametrize("payload", [None, "{bad-json", [], {}, {"dataset": []}])
def test_tool_dataset_attrs_reject_invalid_envelope(payload) -> None:
    ds = xr.Dataset(attrs={_persistence_constants.TOOL_ATTRS_VERSION_ATTR: 1})
    if payload is not None:
        ds.attrs[_persistence_constants.TOOL_ENCODED_ATTRS_ATTR] = (
            payload if isinstance(payload, str) else json.dumps(payload)
        )
    with pytest.raises(ValueError, match=r"[Ii]nvalid.*tool attribute"):
        imagetool_serialization.restore_tool_dataset_attrs(ds)


def test_tool_dataset_attrs_reject_invalid_variable_table() -> None:
    ds = xr.Dataset(
        attrs={
            _persistence_constants.TOOL_ATTRS_VERSION_ATTR: 1,
            _persistence_constants.TOOL_ENCODED_ATTRS_ATTR: json.dumps(
                {"dataset": [], "variables": []}
            ),
        }
    )
    with pytest.raises(TypeError, match="must be a mapping"):
        imagetool_serialization.restore_tool_dataset_attrs(ds)


@pytest.mark.parametrize(
    ("scope", "entries", "error"),
    [
        (None, {}, "must be a list"),
        (None, [[{}]], "Invalid encoded tool attribute entry"),
        (None, [[{}, {"kind": "none"}]], "Invalid encoded tool attribute name"),
        (None, [[{"kind": "str", "value": ""}, {"kind": "none"}]], "attribute name"),
        (None, [[{"kind": "int", "value": 1}, {"kind": "none"}]], "attribute name"),
        (
            None,
            [[{"kind": "str", "value": "new", "extra": 1}, {"kind": "none"}]],
            "attribute name",
        ),
        (None, [[{"kind": "str", "value": "new"}, {}]], "attribute value"),
        (
            None,
            [[{"kind": "str", "value": "new"}, {"kind": "bool", "value": "true"}]],
            "attribute value",
        ),
        ("missing", [], "missing variable"),
        (
            "data",
            [[{"kind": "str", "value": "units"}, {"kind": "none"}]],
            "cannot replace metadata",
        ),
        (
            None,
            [[{"kind": "str", "value": "new"}, {"kind": "none"}]] * 2,
            "cannot replace metadata",
        ),
    ],
)
def test_tool_dataset_attrs_reject_invalid_entries(scope, entries, error) -> None:
    ds = xr.Dataset({"data": ("x", [1.0])})
    ds["data"].attrs["units"] = "counts"
    ds.attrs[_persistence_constants.TOOL_ATTRS_VERSION_ATTR] = 1
    ds.attrs[_persistence_constants.TOOL_ENCODED_ATTRS_ATTR] = json.dumps(
        {
            "dataset": entries if scope is None else [],
            "variables": {} if scope is None else {scope: entries},
        }
    )
    before = ds.copy(deep=True)
    with pytest.raises((TypeError, ValueError), match=error):
        imagetool_serialization.restore_tool_dataset_attrs(ds)
    xr.testing.assert_identical(ds, before)


@pytest.mark.parametrize(
    "key",
    [
        "tool_cls_qualname",
        "tool_state",
        "tool_script_inputs",
        "erlab_code_trust_payload_entries",
    ],
)
def test_tool_dataset_attrs_keep_control_metadata_native(key, monkeypatch) -> None:
    with pytest.raises(TypeError, match="control attribute"):
        imagetool_serialization.prepare_tool_dataset(
            xr.Dataset(attrs={key: {"bad": 1}})
        )

    def fail_if_decoded(value):
        raise AssertionError("Control metadata must be rejected before decoding values")

    monkeypatch.setattr(imagetool_serialization, "_decode_array", fail_if_decoded)
    ds = xr.Dataset(attrs={_persistence_constants.TOOL_ATTRS_VERSION_ATTR: 1})
    ds.attrs[_persistence_constants.TOOL_ENCODED_ATTRS_ATTR] = json.dumps(
        {
            "dataset": [[{"kind": "str", "value": key}, {"kind": "ndarray"}]],
            "variables": {},
        }
    )
    with pytest.raises(ValueError, match="cannot replace metadata"):
        imagetool_serialization.restore_tool_dataset_attrs(ds)


def test_workspace_file_suffix_helpers_collect_nested_inputs(tmp_path) -> None:
    first = _workspace_test_file_spec(tmp_path / "scan_a.h5")
    second = _workspace_test_file_spec(tmp_path / "scan_b.h5")
    third = _workspace_test_file_spec(tmp_path / "scan_c.h5")
    nested = script(
        start_label="Combine",
        seed_code="derived = data_0 + data_1",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_1", label="B", provenance_spec=second),
            ScriptInput(name="data_2", label="C", provenance_spec=third),
            ScriptInput(name="data_0", label="A duplicate", provenance_spec=first),
        ),
    )
    combined = script(
        start_label="Combine nested",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="A", provenance_spec=first),
            ScriptInput(name="nested", label="Nested", provenance_spec=nested),
        ),
    )

    stems = workspace_loading._workspace_provenance_file_stems(combined)

    assert stems == ("scan_a", "scan_b", "scan_c")
    assert (
        workspace_loading._workspace_compact_file_suffix(stems)
        == " (scan_a, scan_b, +1)"
    )


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\name\data\scan.h5",
        r"\\server\share\data\scan.h5",
        "C:/Users/name/data/scan.h5",
        "/Users/name/data/scan.h5",
    ],
)
def test_workspace_file_suffix_helpers_accept_cross_platform_paths(path: str) -> None:
    spec = _workspace_test_file_spec(pathlib.Path(path))

    assert workspace_loading._workspace_provenance_file_stems(spec) == ("scan",)


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"itool_title": "2: manual (scan)", "itool_name": "scan"}, "manual"),
        ({"itool_title": "scan", "itool_name": ""}, None),
        ({"itool_title": "scan (scan)", "itool_name": "scan"}, None),
    ],
)
def test_workspace_legacy_title_migration_ignores_generated_file_labels(
    tmp_path,
    attrs,
    expected,
) -> None:
    ds = xr.Dataset(attrs=attrs)
    spec = _workspace_test_file_spec(tmp_path / "scan.h5")

    assert workspace_loading._legacy_saved_title_data_name(ds, spec) == expected


def test_qt_bytearray_base64_helpers_reject_invalid_values() -> None:
    value = QtCore.QByteArray(b"layout-state")
    encoded = erlab.interactive.utils._qt_bytearray_to_base64(value)

    decoded = erlab.interactive.utils._qt_bytearray_from_base64(encoded)
    assert decoded == value

    assert erlab.interactive.utils._qt_bytearray_from_base64(b"\xff") is None
    assert erlab.interactive.utils._qt_bytearray_from_base64("%%not-base64%%") is None
    assert erlab.interactive.utils._qt_bytearray_from_base64("") is None


def test_qt_window_state_helpers_parse_invalid_and_restore_size(qtbot) -> None:
    assert qt_state.QtWindowState.model_validate({"rect": None}).rect is None
    assert qt_state.QtWindowState.model_validate(
        {"rect": np.asarray([1, 2, 3, 4])}
    ).rect == (1, 2, 3, 4)
    with pytest.raises(pydantic.ValidationError):
        qt_state.QtWindowState.model_validate({"rect": [1, 2, 3]})

    assert qt_state.qt_bytearray_from_base64(object()) is None
    assert qt_state.parse_qt_window_state(b"\xff") is None
    assert qt_state.parse_qt_window_state("{") is None
    assert qt_state.parse_qt_window_state(object()) is None
    assert qt_state.parse_qt_window_state({"rect": [1, 2, 3]}) is None
    assert qt_state.parse_qt_window_state({"rect": "1234"}) is None
    assert qt_state.parse_qt_window_state({"rect": 1}) is None
    assert qt_state.parse_qt_window_state({"rect": [0, 0, None, 45]}) is None
    assert qt_state.parse_qt_window_state('{"rect": [0, 0, Infinity, 45]}') is None

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    assert not qt_state.restore_qt_window_state(widget, "{")
    initial_position = widget.pos()
    assert not widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_Moved)
    assert qt_state.restore_qt_window_state(
        widget, {"geometry": "", "rect": [5000, 6000, 123, 45]}
    )
    assert widget.pos() == initial_position
    assert not widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_Moved)
    assert widget.size() == QtCore.QSize(123, 45)

    widget.show()
    QtWidgets.QApplication.processEvents()
    assert any(
        screen.availableGeometry().intersects(widget.frameGeometry())
        for screen in QtWidgets.QApplication.screens()
    )


def test_qt_window_state_leaves_never_shown_geometry_unset(qtbot) -> None:
    class HintedWindow(QtWidgets.QMainWindow):
        def sizeHint(self) -> QtCore.QSize:
            return QtCore.QSize(321, 243)

    source = HintedWindow()
    restored = HintedWindow()
    qtbot.addWidget(source)
    qtbot.addWidget(restored)
    restored_size = restored.size()

    state = qt_state.qt_window_state(source)

    assert state == qt_state.QtWindowState(visible=False)
    assert qt_state.qt_window_state_payload(source) == {"visible": False}
    assert json.loads(qt_state.qt_window_state_json(source)) == {"visible": False}
    assert not qt_state.restore_qt_window_state(restored, state)
    assert restored.size() == restored_size

    source.show()
    restored.show()
    QtWidgets.QApplication.processEvents()

    assert restored.size() == source.size()


def test_qt_window_state_ignores_native_handle_and_position_before_show(
    qtbot,
) -> None:
    native = QtWidgets.QWidget()
    hidden = QtWidgets.QWidget()
    moved = QtWidgets.QWidget()
    for widget in (native, hidden, moved):
        qtbot.addWidget(widget)

    native.winId()
    hidden.hide()
    moved.move(17, 23)

    for widget in (native, hidden, moved):
        assert qt_state.qt_window_state_payload(widget) == {"visible": False}


def test_qt_window_state_keeps_initialized_hidden_geometry(qtbot) -> None:
    shown = QtWidgets.QWidget()
    resized = QtWidgets.QWidget()
    for widget in (shown, resized):
        qtbot.addWidget(widget)

    shown.show()
    QtWidgets.QApplication.processEvents()
    shown.hide()
    resized.resize(511, 433)

    for widget in (shown, resized):
        state = qt_state.qt_window_state(widget)
        assert state.geometry is not None
        assert state.rect == widget.geometry().getRect()


def test_qt_window_state_native_geometry_is_authoritative(qtbot, monkeypatch) -> None:
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    native_geometry = qt_state.qt_bytearray_to_base64(QtCore.QByteArray(b"native"))
    set_geometry_calls = []

    monkeypatch.setattr(widget, "restoreGeometry", lambda geometry: True)
    monkeypatch.setattr(
        widget,
        "setGeometry",
        lambda *rect: set_geometry_calls.append(rect),
    )

    assert qt_state.restore_qt_window_state(
        widget,
        {"geometry": native_geometry, "rect": [10, 20, 123, 45]},
    )
    assert set_geometry_calls == []


def test_qt_window_state_uses_only_fallback_size_when_native_restore_fails(
    qtbot, monkeypatch
) -> None:
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    native_geometry = qt_state.qt_bytearray_to_base64(QtCore.QByteArray(b"native"))
    resize_calls = []

    monkeypatch.setattr(widget, "restoreGeometry", lambda geometry: False)
    monkeypatch.setattr(
        widget,
        "resize",
        lambda *size: resize_calls.append(size),
    )

    assert qt_state.restore_qt_window_state(
        widget,
        {"geometry": native_geometry, "rect": [5000, 6000, 123, 45]},
    )
    assert resize_calls == [(123, 45)]


@pytest.mark.parametrize(("width", "height"), [(0, 45), (123, -1)])
def test_qt_window_state_rejects_nonpositive_fallback_size(
    qtbot, width: int, height: int
) -> None:
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)

    assert not qt_state.restore_qt_window_state(
        widget, {"rect": [10, 20, width, height]}
    )


def test_qt_window_state_limits_fallback_size_before_calling_qt(
    qtbot, monkeypatch
) -> None:
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    available_size = widget.screen().availableGeometry().size()

    assert qt_state.restore_qt_window_state(widget, {"rect": [10, 20, 2**63, 2**63]})
    assert widget.size() == available_size

    widget.setMaximumSize(200, 150)
    monkeypatch.setattr(widget, "screen", lambda: None)
    assert qt_state.restore_qt_window_state(widget, {"rect": [10, 20, 2**63, 2**63]})
    assert widget.size() == QtCore.QSize(200, 150)


def test_qt_window_state_restores_maximized_normal_geometry(qtbot) -> None:
    screen = QtWidgets.QApplication.primaryScreen()
    assert screen is not None
    available = screen.availableGeometry()
    requested_geometry = QtCore.QRect(
        available.left() + 20,
        available.top() + 20,
        min(320, available.width() - 40),
        min(240, available.height() - 40),
    )

    source = QtWidgets.QMainWindow()
    restored = QtWidgets.QMainWindow()
    qtbot.addWidget(source)
    qtbot.addWidget(restored)
    source.setGeometry(requested_geometry)
    source.show()
    QtWidgets.QApplication.processEvents()
    normal_geometry = source.geometry()
    source.showMaximized()
    QtWidgets.QApplication.processEvents()

    state = qt_state.qt_window_state(source)
    assert qt_state.restore_qt_window_state(restored, state)
    restored.show()
    QtWidgets.QApplication.processEvents()

    assert restored.isMaximized()
    restored_normal_geometry = restored.normalGeometry()
    assert restored_normal_geometry.size() == normal_geometry.size()
    restored.showNormal()
    QtWidgets.QApplication.processEvents()
    assert restored.geometry() == restored_normal_geometry
    assert any(
        candidate.availableGeometry().intersects(restored.frameGeometry())
        for candidate in QtWidgets.QApplication.screens()
    )


def test_imagetool_private_coord_serialization_edge_cases() -> None:
    private_attr = imagetool_serialization.PRIVATE_COORDS_ATTR
    private_prefix = "__erlab_imagetool_coord_"
    data_name = _persistence_constants.ITOOL_DATA_NAME
    valid_payload = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["x"]}]
    )

    assert imagetool_serialization.private_coord_records_from_attrs(
        {private_attr: valid_payload.encode()}
    ) == ({"coord_name": "Fake Motor", "variable_name": "private", "dims": ("x",)},)
    assert (
        imagetool_serialization.private_coord_records_from_attrs({private_attr: 1})
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: "{not-json"}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: json.dumps([[]])}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: json.dumps([{"coord_name": "Fake Motor", "dims": ["x"]}])}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_variable_names(
            xr.Dataset({"other": ("x", [1.0])})
        )
        == ()
    )

    ds = xr.Dataset(
        {
            data_name: ("x", np.arange(2.0)),
            f"{private_prefix}0": ("x", np.arange(2.0) + 10.0),
        },
        coords={"x": np.arange(2.0), "Fake Motor": ("x", np.arange(2.0) + 20.0)},
    )
    encoded = imagetool_serialization.encode_private_coords(ds)

    assert imagetool_serialization.private_coord_variable_names(encoded) == (
        f"{private_prefix}1",
    )
    restored = imagetool_serialization.restore_private_coords(encoded)
    xr.testing.assert_equal(restored.coords["Fake Motor"], ds.coords["Fake Motor"])


def test_imagetool_private_coord_restore_ignores_invalid_records() -> None:
    private_attr = imagetool_serialization.PRIVATE_COORDS_ATTR
    data_name = _persistence_constants.ITOOL_DATA_NAME
    missing_data = xr.Dataset({"other": ("x", [1.0])})

    assert imagetool_serialization.restore_private_coords(missing_data) is missing_data

    payload = json.dumps(
        [
            {"coord_name": "Missing", "variable_name": "missing", "dims": ["x"]},
            {"coord_name": "Bad Dims", "variable_name": "present", "dims": ["z"]},
        ]
    )
    encoded = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "present": ("z", [2.0]),
        },
        attrs={"root": "kept"},
    )
    encoded[data_name].attrs[private_attr] = payload

    restored = imagetool_serialization.restore_private_coords(encoded)

    assert private_attr not in restored[data_name].attrs
    assert "Missing" not in restored.coords
    assert "Bad Dims" not in restored.coords
    assert "present" in restored.data_vars

    legacy = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "plain": ("x", [2.0]),
            "Fake Motor": ("z", [3.0]),
        }
    )

    legacy_restored = imagetool_serialization.restore_private_coords(legacy)

    assert "plain" in legacy_restored.data_vars
    assert "Fake Motor" in legacy_restored.data_vars


def test_workspace_attr_native_detection_handles_edge_types() -> None:
    assert imagetool_serialization.attr_value_writes_natively(b"ok")
    assert not imagetool_serialization.attr_value_writes_natively(b"\xff")
    assert not imagetool_serialization.attr_value_writes_natively(b"a\x00")
    assert imagetool_serialization.attr_value_writes_natively(
        np.array([1, 2], dtype=np.int16)
    )
    assert not imagetool_serialization.attr_value_writes_natively(
        np.array([object()], dtype=object)
    )
    assert imagetool_serialization.attr_value_writes_natively(np.float64(1.0))
    assert not imagetool_serialization.attr_value_writes_natively(
        np.datetime64("2024-01-01")
    )
    assert imagetool_serialization.attr_value_writes_natively(("left", "right"))
    assert imagetool_serialization.attr_value_writes_natively((b"left", b"right"))
    assert imagetool_serialization.attr_value_writes_natively(
        (np.bool_(True), complex(1.0, 2.0))
    )
    assert not imagetool_serialization.attr_value_writes_natively([1, "text"])
    assert not imagetool_serialization.attr_value_writes_natively(("text", b"bytes"))
    assert not imagetool_serialization.attr_value_writes_natively(([1],))


def test_workspace_mixed_scalar_attrs_use_typed_encoding() -> None:
    attrs = {
        "mixed_list": [1, "text"],
        "mixed_tuple": ("text", b"bytes"),
        "native_numbers": [1, 2.0],
    }

    serializable = workspace_format._workspace_serializable_attrs(attrs)

    assert "mixed_list" not in serializable
    assert "mixed_tuple" not in serializable
    assert serializable["native_numbers"] == [1, 2.0]
    restored = workspace_format._restore_workspace_serialized_attrs(serializable)
    assert restored["mixed_list"] == [1, "text"]
    assert restored["mixed_tuple"] == ("text", b"bytes")
    assert restored["native_numbers"] == [1, 2.0]


@pytest.mark.parametrize("dtype", ["S1", "<U1", ">U1"])
def test_workspace_legacy_empty_numpy_string_attrs_roundtrip(dtype, tmp_path) -> None:
    # Released writers expanded empty scalars to one zero code unit.
    scalar = {
        "kind": "numpy_scalar",
        "dtype": dtype,
        "shape": [],
        "data": "AA==" if dtype == "S1" else "AAAAAA==",
    }
    entries = [
        [
            {"kind": "str", "value": "metadata"},
            {
                "kind": "dict",
                "items": [[{"kind": "str", "value": "empty"}, scalar]],
            },
        ]
    ]
    ds = xr.Dataset(
        attrs={
            _persistence_constants.WORKSPACE_ENCODED_ATTRS_ATTR: json.dumps(
                {"version": 1, "attrs": entries}
            )
        }
    )
    path = tmp_path / "legacy-attrs.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    opened = xr.load_dataset(path, engine="h5netcdf")
    expected = np.bytes_(b"") if dtype == "S1" else np.str_("")

    for attrs in (
        workspace_format._restore_workspace_serialized_attrs(opened.attrs),
        workspace_format._restore_workspace_manifest_attrs(entries),
    ):
        _assert_tool_attr_equal(attrs["metadata"]["empty"], expected)
        assert not attrs["metadata"]["empty"]
        # A new save must retain the old value after conversion to the new record.
        restored = workspace_format._restore_workspace_manifest_attrs(
            workspace_format._workspace_manifest_attrs(attrs)
        )
        _assert_tool_attr_equal(restored["metadata"]["empty"], expected)


@pytest.mark.parametrize(
    "key",
    [
        np.str_(""),
        np.bytes_(b""),
        np.str_("β\x00\x00"),
        np.bytes_(b"\xff\x00\x00"),
        np.datetime64("2025-01-01", "ns"),
        np.timedelta64(3, "ns"),
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ],
)
def test_workspace_numpy_keys_preserve_type_and_contents(key) -> None:
    decoded = imagetool_serialization.decode_attr_key(
        imagetool_serialization.encode_attr_key(key)
    )
    _assert_tool_attr_equal(decoded, key)


@pytest.mark.parametrize("key", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_tool_dataset_attrs_keep_nat_keys_distinct_from_none(key) -> None:
    ds = xr.Dataset(attrs={"metadata": {None: "none", key: "nat"}})
    for _ in range(2):
        prepared = imagetool_serialization.prepare_tool_dataset(ds)
        raw = prepared.to_netcdf(engine="h5netcdf", invalid_netcdf=True)
        opened = xr.load_dataset(memoryview(raw), engine="h5netcdf")
        ds = imagetool_serialization.restore_tool_dataset_attrs(opened)
        metadata = ds.attrs["metadata"]
        assert len(metadata) == 2
        assert metadata[None] == "none"
        restored_key = next(item for item in metadata if item is not None)
        _assert_tool_attr_equal(restored_key, key)
        assert metadata[restored_key] == "nat"


@pytest.mark.parametrize("tuple_keys", [False, True])
def test_tool_dataset_attrs_preserve_distinct_nan_keys(tuple_keys) -> None:
    keys = [float("nan"), float("nan")]
    if tuple_keys:
        keys = [(key,) for key in keys]
    ds = xr.Dataset(attrs={"metadata": dict(zip(keys, [1, 2], strict=True))})
    for _ in range(2):
        prepared = imagetool_serialization.prepare_tool_dataset(ds)
        raw = prepared.to_netcdf(engine="h5netcdf", invalid_netcdf=True)
        opened = xr.load_dataset(memoryview(raw), engine="h5netcdf")
        ds = imagetool_serialization.restore_tool_dataset_attrs(opened)
        metadata = ds.attrs["metadata"]
        assert len(metadata) == 2
        assert list(metadata.values()) == [1, 2]
        assert all(np.isnan(key[0] if tuple_keys else key) for key in metadata)


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_workspace_attr_numpy_unicode_scalar_byteorder(byteorder) -> None:
    text = "\ufeffβ😀\ud800\x00\x00"
    array = np.array(text, dtype=f"{byteorder}U{len(text)}")
    payload = imagetool_serialization.encode_attr_value(array)
    payload["kind"] = "numpy_string"

    decoded = imagetool_serialization.decode_attr_value(payload)

    _assert_tool_attr_equal(decoded, np.str_(text))


@pytest.mark.parametrize("kind", ["numpy_scalar", "numpy_string"])
@pytest.mark.parametrize("dtype", ["U0", "S0"])
def test_workspace_attr_zero_width_strings_reject_nonempty_data(dtype, kind) -> None:
    payload = {"kind": kind, "dtype": dtype, "shape": [], "data": "AA=="}

    with pytest.raises(ValueError, match="zero-width string array must have no data"):
        imagetool_serialization.decode_attr_value(payload)


@pytest.mark.parametrize("kind", ["numpy_scalar", "numpy_string"])
@pytest.mark.parametrize("dtype", ["U1", "S1", "i4"])
def test_workspace_attr_numpy_scalar_rejects_array_shape(dtype, kind) -> None:
    payload = imagetool_serialization.encode_attr_value(np.zeros(1, dtype=dtype))
    payload["kind"] = kind

    with pytest.raises(ValueError, match="scalar must have an empty shape"):
        imagetool_serialization.decode_attr_value(payload)


def test_workspace_attr_numpy_string_rejects_nonstring_dtype() -> None:
    payload = imagetool_serialization.encode_attr_value(np.float64(1.0))
    payload["kind"] = "numpy_string"

    with pytest.raises(TypeError, match="string must have a string dtype"):
        imagetool_serialization.decode_attr_value(payload)


def test_workspace_attr_typed_encoding_roundtrips_safe_values(caplog) -> None:
    import decimal
    import math

    value = {
        None: None,
        False: True,
        np.int64(3): 5,
        7: np.float64(2.5),
        1.5: math.inf,
        complex(1.0, -2.0): -math.inf,
        "nan": math.nan,
        b"\xff": b"\x00\xff",
        ("tuple", 2): [
            np.array([[1, 2], [3, 4]], dtype=np.int16),
            np.array([{"nested": (None, complex(3.0, 4.0))}], dtype=object),
        ],
    }

    decoded = imagetool_serialization.decode_attr_value(
        imagetool_serialization.encode_attr_value(value)
    )

    assert decoded[None] is None
    assert decoded[False] is True
    assert decoded[3] == 5
    assert decoded[7] == np.float64(2.5)
    assert decoded[1.5] == math.inf
    assert decoded[complex(1.0, -2.0)] == -math.inf
    assert math.isnan(decoded["nan"])
    assert decoded[b"\xff"] == b"\x00\xff"
    np.testing.assert_array_equal(
        decoded[("tuple", 2)][0], np.array([[1, 2], [3, 4]], dtype=np.int16)
    )
    assert decoded[("tuple", 2)][1][0]["nested"] == (None, complex(3.0, 4.0))

    with pytest.raises(TypeError, match="unsupported attr key type"):
        imagetool_serialization.encode_attr_key(["bad"])
    with pytest.raises(TypeError, match="unsupported numeric attr type"):
        imagetool_serialization.encode_attr_value(decimal.Decimal("1.0"))
    with pytest.raises(TypeError, match="must be a mapping"):
        imagetool_serialization.decode_attr_value([])
    with pytest.raises(TypeError, match="unknown workspace attr value kind"):
        imagetool_serialization.decode_attr_value({"kind": "unknown"})
    with pytest.raises(TypeError, match="not hashable"):
        imagetool_serialization.decode_attr_key({"kind": "list", "items": []})

    assert workspace_format._workspace_encoded_attr_entries(b"\xff") is None
    assert workspace_format._workspace_encoded_attr_entries(1) is None
    assert workspace_format._workspace_encoded_attr_entries("{bad-json") is None
    assert (
        workspace_format._workspace_encoded_attr_entries(
            json.dumps({"version": -1, "attrs": []})
        )
        is None
    )
    assert (
        workspace_format._workspace_encoded_attr_entries(
            json.dumps(
                {
                    "version": _persistence_constants.WORKSPACE_ENCODED_ATTRS_VERSION,
                    "attrs": [["too-short"]],
                }
            )
        )
        is None
    )

    invalid_payload = json.dumps(
        {
            "version": _persistence_constants.WORKSPACE_ENCODED_ATTRS_VERSION,
            "attrs": [[{"kind": "list", "items": []}, {"kind": "str", "value": "x"}]],
        }
    )
    with caplog.at_level(logging.WARNING):
        restored = workspace_format._restore_workspace_serialized_attrs(
            {_persistence_constants.WORKSPACE_ENCODED_ATTRS_ATTR: invalid_payload}
        )
    assert restored == {}
    assert "Ignoring invalid encoded workspace attribute" in caplog.text


def test_workspace_metadata_helpers_cover_invalid_payloads() -> None:
    manifest = workspace_format._workspace_manifest_payload(
        root_order=["1"],
        nodes=[{"path": "1"}],
        erlab_version="test",
    )
    raw_manifest = json.dumps(manifest)

    encoded_manifest = workspace_format._workspace_manifest_from_attrs(
        {_persistence_constants.WORKSPACE_MANIFEST_ATTR: raw_manifest.encode()}
    )
    assert encoded_manifest["root_order"] == ["1"]
    decoded_manifest = workspace_format._workspace_manifest_from_attrs(
        {_persistence_constants.WORKSPACE_MANIFEST_ATTR: raw_manifest}
    )
    assert decoded_manifest["nodes"] == [{"path": "1"}]
    assert (
        workspace_format._workspace_manifest_from_attrs(
            {_persistence_constants.WORKSPACE_MANIFEST_ATTR: "{not-json"}
        )
        == {}
    )

    assert list(workspace_format._iter_workspace_manifest_node_entries(None)) == []
    assert (
        list(
            workspace_format._iter_workspace_manifest_node_entries({"nodes": "invalid"})
        )
        == []
    )
    assert workspace_format._workspace_manifest_payload_entries(
        {
            "nodes": [
                {"uid": "legacy", "kind": "tool", "path": "2"},
                {"uid": "missing", "kind": "tool"},
            ]
        }
    ) == [("legacy", "tool", "2/tool")]
    assert (
        workspace_format._workspace_manifest_payload_path(manifest, "missing") is None
    )


def test_workspace_embedded_script_entry_uses_exact_filename_and_hash() -> None:
    source_hash = "a" * 64
    entry = workspace_format._WorkspaceEmbeddedScriptEntry(
        script_name="Gaussian_Tools.PY",
        source_hash=source_hash,
        object_id=f"extension-source-{source_hash}",
    )

    assert entry.script_name == "Gaussian_Tools.PY"
    assert _script_name_key(entry.script_name) == "gaussian_tools.py"
    for script_name in ("", "nested/script.py", "bad\\script.py", "bad\x00.py"):
        with pytest.raises(pydantic.ValidationError):
            workspace_format._WorkspaceEmbeddedScriptEntry(
                script_name=script_name,
                source_hash=source_hash,
                object_id=f"extension-source-{source_hash}",
            )
    with pytest.raises(pydantic.ValidationError, match="does not match"):
        workspace_format._WorkspaceEmbeddedScriptEntry(
            script_name="valid.py",
            source_hash=source_hash,
            object_id="extension-source-wrong",
        )


def test_workspace_immutable_generation_helpers_filter_invalid_manifest_entries() -> (
    None
):
    manifest = {
        "nodes": [
            {"path": "0", "kind": "imagetool", "payload_object_id": "image"},
            {"path": "/1/", "kind": "tool", "payload_object_id": "tool"},
            {"path": "2", "kind": "unknown", "payload_object_id": "ignored"},
            {"path": 3, "kind": "tool", "payload_object_id": "ignored"},
            {"path": "4", "kind": "tool", "payload_object_id": ""},
        ]
    }

    assert workspace_format._workspace_manifest_legacy_reader_rebindings(manifest) == {
        "/0/imagetool": "image",
        "/1/tool": "tool",
    }
    assert workspace_format._workspace_schema_uses_immutable_generations(5)
    assert workspace_format._workspace_schema_uses_immutable_generations(6)
    assert not workspace_format._workspace_schema_uses_immutable_generations(4)
    assert not workspace_format._workspace_schema_uses_immutable_generations(7)


def test_workspace_manifest_attrs_reject_invalid_entries(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        encoded = workspace_format._workspace_manifest_attrs(
            {"kept": 1, "dropped": object()}
        )
    assert workspace_format._restore_workspace_manifest_attrs(encoded) == {"kept": 1}
    assert "Dropping workspace attribute" in caplog.text

    with pytest.raises(TypeError, match="must be a list"):
        workspace_format._restore_workspace_manifest_attrs({})
    with pytest.raises(TypeError, match="entry is invalid"):
        workspace_format._restore_workspace_manifest_attrs([["too-short"]])
