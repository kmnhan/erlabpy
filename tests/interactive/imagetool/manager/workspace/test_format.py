import json
import logging

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


def test_qt_window_state_helpers_parse_invalid_and_restore_rect(qtbot) -> None:
    assert qt_state.QtWindowState.model_validate({"rect": None}).rect is None
    with pytest.raises(pydantic.ValidationError):
        qt_state.QtWindowState.model_validate({"rect": [1, 2, 3]})

    assert qt_state.qt_bytearray_from_base64(object()) is None
    assert qt_state.parse_qt_window_state(b"\xff") is None
    assert qt_state.parse_qt_window_state("{") is None
    assert qt_state.parse_qt_window_state({"rect": [1, 2, 3]}) is None

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    assert qt_state.restore_qt_window_state(
        widget, {"geometry": "", "rect": [10, 20, 123, 45]}
    )
    assert widget.geometry().getRect() == (10, 20, 123, 45)


def test_imagetool_private_coord_serialization_edge_cases() -> None:
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    private_prefix = imagetool_serialization._PRIVATE_COORD_VAR_PREFIX
    data_name = imagetool_serialization.ITOOL_DATA_NAME
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
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    data_name = imagetool_serialization.ITOOL_DATA_NAME
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
    assert workspace_format._workspace_attr_value_writes_natively(b"ok")
    assert not workspace_format._workspace_attr_value_writes_natively(b"\xff")
    assert not workspace_format._workspace_attr_value_writes_natively(b"a\x00")
    assert workspace_format._workspace_attr_value_writes_natively(
        np.array([1, 2], dtype=np.int16)
    )
    assert not workspace_format._workspace_attr_value_writes_natively(
        np.array([object()], dtype=object)
    )
    assert workspace_format._workspace_attr_value_writes_natively(np.float64(1.0))
    assert not workspace_format._workspace_attr_value_writes_natively(
        np.datetime64("2024-01-01")
    )
    assert workspace_format._workspace_attr_value_writes_natively(("left", "right"))
    assert workspace_format._workspace_attr_value_writes_natively((b"left", b"right"))
    assert workspace_format._workspace_attr_value_writes_natively(
        (np.bool_(True), complex(1.0, 2.0))
    )
    assert not workspace_format._workspace_attr_value_writes_natively([1, "text"])
    assert not workspace_format._workspace_attr_value_writes_natively(
        ("text", b"bytes")
    )
    assert not workspace_format._workspace_attr_value_writes_natively(([1],))


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

    decoded = workspace_format._workspace_decode_attr_value(
        workspace_format._workspace_encode_attr_value(value)
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
        workspace_format._workspace_encode_attr_key(["bad"])
    with pytest.raises(TypeError, match="unsupported numeric attr type"):
        workspace_format._workspace_encode_attr_value(decimal.Decimal("1.0"))
    with pytest.raises(TypeError, match="must be a mapping"):
        workspace_format._workspace_decode_attr_value([])
    with pytest.raises(TypeError, match="unknown workspace attr value kind"):
        workspace_format._workspace_decode_attr_value({"kind": "unknown"})
    with pytest.raises(TypeError, match="not hashable"):
        workspace_format._workspace_decode_attr_key({"kind": "list", "items": []})

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
                    "version": workspace_format._WORKSPACE_ENCODED_ATTRS_VERSION,
                    "attrs": [["too-short"]],
                }
            )
        )
        is None
    )

    invalid_payload = json.dumps(
        {
            "version": workspace_format._WORKSPACE_ENCODED_ATTRS_VERSION,
            "attrs": [[{"kind": "list", "items": []}, {"kind": "str", "value": "x"}]],
        }
    )
    with caplog.at_level(logging.WARNING):
        restored = workspace_format._restore_workspace_serialized_attrs(
            {workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR: invalid_payload}
        )
    assert restored == {}
    assert "Ignoring invalid encoded workspace attribute" in caplog.text


def test_workspace_metadata_helpers_cover_invalid_payloads() -> None:
    manifest_attrs = workspace_format._workspace_root_attrs_payload(
        root_order=["1"],
        nodes=[{"path": "1"}],
        delta_save_count=2,
        erlab_version="test",
    )
    raw_manifest = manifest_attrs[workspace_format._WORKSPACE_MANIFEST_ATTR]

    assert (
        workspace_format._workspace_manifest_from_attrs(
            {workspace_format._WORKSPACE_MANIFEST_ATTR: raw_manifest.encode()}
        )["delta_save_count"]
        == 2
    )
    manifest = workspace_format._workspace_manifest_from_attrs(
        {workspace_format._WORKSPACE_MANIFEST_ATTR: raw_manifest}
    )
    assert workspace_format._workspace_manifest_repack_estimate(
        manifest, delta_save_count=2
    ) == (0, 0, True)
    assert workspace_format._workspace_manifest_repack_estimate(
        {"delta_save_count": 2}, delta_save_count=2
    ) == (0, 0, False)
    assert workspace_format._workspace_manifest_repack_estimate(
        None, delta_save_count=2
    ) == (0, 0, False)
    assert (
        workspace_format._workspace_manifest_nonnegative_int(
            {"estimated_obsolete_bytes": "not-an-int"},
            "estimated_obsolete_bytes",
        )
        == 0
    )
    assert (
        workspace_format._workspace_manifest_from_attrs(
            {workspace_format._WORKSPACE_MANIFEST_ATTR: "{not-json"}
        )
        == {}
    )
    assert (
        workspace_format._workspace_delta_save_count_from_attrs(
            {
                workspace_format._WORKSPACE_MANIFEST_ATTR: (
                    '{"delta_save_count": "not-an-int"}'
                )
            }
        )
        == 0
    )
    with pytest.raises(ValueError, match="current workspace schema"):
        workspace_format._compacted_workspace_root_attrs(
            {"imagetool_workspace_schema_version": 1}
        )
    assert workspace_format._workspace_root_attrs_with_repack_estimate(
        {"imagetool_workspace_schema_version": 1},
        estimated_obsolete_bytes=1,
        replacement_delta_count=1,
    ) == {"imagetool_workspace_schema_version": 1}
