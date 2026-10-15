"""Source lifecycle tests independent of the registered payload type."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import xarray as xr
from qtpy import QtWidgets

from erlab.interactive._figurecomposer import _sources
from erlab.interactive._figurecomposer._model._document import FigureDocument
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
)
from erlab.interactive._figurecomposer._sources import (
    ArraySourcePayload,
    EncodedSourcePayload,
    OpaqueSourcePayload,
    SourceSpec,
    decode_source_payload,
    encode_source_payload,
    register_source_spec,
    source_payload_kind,
    source_spec_for_kind,
)
from erlab.interactive._figurecomposer._tool import FigureComposerTool

from ._common import _restored_figure_composer_from_netcdf


@dataclasses.dataclass(frozen=True)
class _CurvePayload:
    data: xr.DataArray
    unit: str


@pytest.fixture
def curve_type(monkeypatch):
    monkeypatch.setattr(_sources, "_SOURCE_SPECS", dict(_sources._SOURCE_SPECS))
    register_source_spec(
        SourceSpec(
            kind="test-curve",
            label="Curve",
            payload_type=_CurvePayload,
            duplicate=lambda payload: dataclasses.replace(
                payload, data=payload.data.copy(deep=False)
            ),
            encode=lambda payload: EncodedSourcePayload(
                {"unit": payload.unit}, {"curve": payload.data}
            ),
            decode=lambda encoded: _CurvePayload(
                encoded.arrays["curve"], encoded.metadata["unit"]
            ),
            safe_value=lambda payload: payload.data,
        )
    )
    return _CurvePayload(xr.DataArray(np.arange(4.0), dims="x"), "eV")


def test_third_source_type_lifecycle_and_codec(curve_type):
    source = FigureSourceState(
        name="curve",
        kind="test-curve",
        provider_id="curve-provider",
        node_uid="tool-node",
        provider_selection={"output": "line"},
    )
    changes = []
    document = FigureDocument(
        FigureRecipeState(sources=(source,), primary_source="curve"),
        source_payloads={"curve": curve_type},
        changed_callback=lambda recipe, payloads: changes.append((recipe, payloads)),
    )
    assert document.source_kind_names("test-curve") == ("curve",)
    assert document.source_data == {}
    assert document.source_payload("curve", _CurvePayload) is curve_type
    assert document.source_payload("curve", ArraySourcePayload) is None
    assert document.rename_source("curve", "renamed")
    assert document.recipe.primary_source == "renamed"
    assert document.source_payloads == {"renamed": curve_type}
    assert document.duplicate_sources(("renamed",)) == ("renamed_copy",)
    duplicate = document.source_payload("renamed_copy", _CurvePayload)
    assert duplicate is not None
    assert duplicate is not curve_type
    xr.testing.assert_identical(duplicate.data, curve_type.data)
    encoded = encode_source_payload(duplicate)
    assert encoded.metadata == {"unit": "eV"}
    assert encoded.arrays["curve"] is duplicate.data
    restored = decode_source_payload("test-curve", encoded)
    assert isinstance(restored, _CurvePayload)
    xr.testing.assert_identical(restored.data, curve_type.data)
    assert document.remove_sources(("renamed",)) == ("renamed",)
    assert document.recipe.primary_source == "renamed_copy"
    assert set(document.source_payloads) == {"renamed_copy"}
    assert changes == [(True, True)] * 3


def test_third_source_type_add_refresh_and_paste(curve_type):
    source = FigureSourceState(
        name="curve",
        kind="test-curve",
        node_uid="node",
        provider_id="curve-provider",
        provider_selection={"output": "line"},
    )
    document = FigureDocument(FigureRecipeState())
    assert document.add_source_payloads((source,), {"curve": curve_type}).added == (
        ("curve", "curve"),
    )
    replacement = dataclasses.replace(curve_type, unit="meV")
    renamed_input = source.model_copy(update={"name": "new_name"})
    result = document.add_source_payloads((renamed_input,), {"new_name": replacement})
    assert result.updated == (("new_name", "curve"),)
    assert document.source_payloads["curve"] is replacement
    assert document.update_source_payloads({"curve": curve_type}).updated == ("curve",)
    operation = FigureOperationState.custom(label="curve", code="pass").model_copy(
        update={"sources": ("curve",)}
    )
    pasted = document.paste_operations(
        0, (operation,), (source,), source_payloads={"curve": curve_type}
    )
    assert pasted.source_data_changed
    assert document.recipe.operations[0].sources == ("curve_copy",)
    assert isinstance(document.source_payloads["curve_copy"], _CurvePayload)
    assert len(document.source_payloads) == 2


def test_unknown_source_type_is_preserved_without_importing_code():
    data = xr.DataArray(np.arange(3.0), dims="x")
    encoded = EncodedSourcePayload(
        {"module": "unavailable.module", "options": {"value": 2}}, {"values": data}
    )
    restored = decode_source_payload("future-source", encoded)
    assert isinstance(restored, OpaqueSourcePayload)
    assert source_payload_kind(restored) == "future-source"
    assert encode_source_payload(restored) is encoded
    document = FigureDocument(
        FigureRecipeState(
            sources=(FigureSourceState(name="future", kind="future-source"),)
        ),
        source_payloads={"future": restored},
    )
    assert document.rename_source("future", "saved")
    assert document.duplicate_sources(("saved",)) == ("saved_copy",)
    copied = encode_source_payload(document.source_payloads["saved_copy"])
    assert copied.metadata == encoded.metadata
    copied.metadata["options"]["value"] = 3
    assert encoded.metadata["options"]["value"] == 2


def test_fit_decoding_remains_opaque(monkeypatch):
    from erlab.interactive._figurecomposer._fit_sources import FigureFitSourceSnapshot

    def fail_if_executed(*args, **kwargs):
        raise AssertionError("Decoding an opaque payload must not execute a model")

    monkeypatch.setattr(FigureFitSourceSnapshot, "result_dataset", fail_if_executed)
    snapshot = FigureFitSourceSnapshot(b"opaque model bytes")
    decoded = decode_source_payload("fit", encode_source_payload(snapshot))
    assert isinstance(decoded, FigureFitSourceSnapshot)
    assert decoded.blob == snapshot.blob


def test_legacy_fit_source_storage_remains_readable(qtbot):
    import json

    import lmfit

    from erlab.interactive._figurecomposer._fit_sources import FigureFitSourceSnapshot
    from erlab.interactive.utils import ToolWindow

    data = xr.DataArray(np.arange(7.0), dims="x", coords={"x": np.arange(7.0)})
    snapshot = FigureFitSourceSnapshot.from_dataset(
        data.xlm.modelfit("x", lmfit.models.LinearModel())
    )
    tool = FigureComposerTool.from_source_payloads(
        {"fit": snapshot},
        sources=(FigureSourceState(name="fit", kind="fit"),),
        operations=(),
    )
    qtbot.addWidget(tool)
    saved = tool.to_dataset()
    saved.attrs.pop("figure_composer_source_manifest")
    saved.attrs["figure_composer_fit_sources"] = json.dumps(
        {"fit": snapshot.to_payload()}
    )
    restored = ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    captured = restored._document.source_payload("fit", FigureFitSourceSnapshot)
    assert captured is not None
    assert captured.blob == snapshot.blob
    assert restored.source_data() == {}


def test_array_codec_keeps_native_data_and_selection_base():
    data = xr.DataArray(np.arange(3.0), dims="x")
    base = xr.DataArray(np.arange(6.0), dims="x")
    decoded = decode_source_payload(
        "array", encode_source_payload(ArraySourcePayload(data, base))
    )
    assert isinstance(decoded, ArraySourcePayload)
    assert decoded.data is data
    assert decoded.selection_base is base
    assert source_spec_for_kind("array").supports_selection


def test_array_selection_adapter_preserves_other_source_types(curve_type):
    data = xr.DataArray(np.arange(3.0), dims="x")
    document = FigureDocument(
        FigureRecipeState(), source_payloads={"curve": curve_type}
    )
    document.replace_source_payloads({"array": data}, {})
    assert document.source_payloads["curve"] is curve_type
    assert document.source_data["array"] is data
    document.replace_source_payloads({}, {})
    assert dict(document.source_payloads) == {"curve": curve_type}


def test_complete_array_snapshot_retains_selection_base():
    base = xr.DataArray(np.arange(6.0), dims="x")
    data = base.isel(x=slice(1, 4))
    source = FigureSourceState(name="selection", isel={"x": slice(1, 4)})
    payload = ArraySourcePayload(data, base)
    document = FigureDocument(FigureRecipeState())
    result = document.add_source_payloads((source,), {"selection": payload})
    assert result.added == (("selection", "selection"),)
    assert document.source_payloads["selection"] is payload
    assert document.source_data["selection"] is data
    assert document.source_selection_base_data["selection"] is base


@pytest.mark.parametrize("primary_source", ["curve", "array"])
def test_third_source_type_tool_roundtrip(curve_type, qtbot, tmp_path, primary_source):
    curve_type.data.name = "captured_curve"
    curve_type.data.attrs["units"] = "counts"
    array = xr.DataArray(np.arange(7.0), dims="energy", name="measurement")
    tool = FigureComposerTool.from_source_payloads(
        {"curve": curve_type, "array": ArraySourcePayload(array)},
        sources=(
            FigureSourceState(name="curve", kind="test-curve"),
            FigureSourceState(name="array"),
        ),
        primary_source=primary_source,
        operations=(),
    )
    qtbot.addWidget(tool)
    saved = tool.to_dataset()
    assert "figure_composer_source_manifest" in saved.attrs
    assert "figure_composer_fit_sources" not in saved.attrs
    restored = _restored_figure_composer_from_netcdf(tool, qtbot, tmp_path)
    assert set(restored.source_payloads()) == {"curve", "array"}
    captured = restored._document.source_payload("curve", _CurvePayload)
    assert captured is not None
    assert captured.unit == curve_type.unit
    xr.testing.assert_identical(captured.data, curve_type.data)
    xr.testing.assert_identical(restored.source_data()["array"], array)
    assert restored.tool_status.primary_source == primary_source


def test_third_source_type_payload_refresh_undo_redo(curve_type, qtbot):
    source = FigureSourceState(name="curve", kind="test-curve")
    tool = FigureComposerTool.from_source_payloads(
        {"curve": curve_type}, sources=(source,), operations=()
    )
    qtbot.addWidget(tool)
    tool._reset_history_stack()
    refreshed = dataclasses.replace(curve_type, data=curve_type.data + 10)
    assert tool.replace_source_payload("curve", source, refreshed)
    xr.testing.assert_identical(tool._source_value("curve"), refreshed.data)
    tool._undo_recorded_state()
    xr.testing.assert_identical(tool._source_value("curve"), curve_type.data)
    tool._redo_recorded_state()
    xr.testing.assert_identical(tool._source_value("curve"), refreshed.data)


def test_legacy_provider_state_migrates_without_fit_specific_fields():
    fit = FigureSourceState.model_validate(
        {
            "name": "fit",
            "kind": "fit",
            "fit_selection": {"index": 2},
        }
    )
    assert fit.provider_id == "fit"
    assert fit.provider_selection == {"index": 2}
    assert "fit_selection" not in fit.model_dump()
    array = FigureSourceState(name="array", node_uid="node", data_role="source")
    assert array.provider_id == "array"
    assert array.provider_selection == {"data_role": "source"}
    assert array.use_node_data


def test_tool_provider_array_does_not_reference_node_data():
    source = FigureSourceState(
        name="derived", node_uid="node", provider_id="derived", use_node_data=False
    )
    assert not source.use_node_data
    assert source.to_script_input() is None
    restored = FigureSourceState.model_validate_json(source.model_dump_json())
    assert not restored.use_node_data
    legacy = FigureSourceState(name="image", node_uid="node")
    script_input = legacy.to_script_input()
    assert script_input is not None
    assert script_input.node_uid == "node"


def test_executable_source_native_slots_are_inspected_without_materializing(
    monkeypatch, qtbot
):
    from erlab.interactive._code_trust import create_payload_entry

    @dataclasses.dataclass(frozen=True)
    class ExecutableCurve:
        data: xr.DataArray
        code: str

    def fail_if_executed(payload):
        raise AssertionError("Trust inspection must not materialize the source")

    monkeypatch.setattr(_sources, "_SOURCE_SPECS", dict(_sources._SOURCE_SPECS))
    register_source_spec(
        SourceSpec(
            kind="test-executable",
            payload_type=ExecutableCurve,
            duplicate=lambda payload: payload,
            encode=lambda payload: EncodedSourcePayload(
                {"code": payload.code}, {"curve": payload.data}
            ),
            decode=lambda encoded: ExecutableCurve(
                encoded.arrays["curve"], encoded.metadata["code"]
            ),
            safe_value=lambda payload: payload.data,
            materialize=fail_if_executed,
            trust_entries=lambda payload, alias, prefix: (
                create_payload_entry(
                    "test.executable-source",
                    f"{prefix}/sources/{alias}",
                    payload.code,
                    payload.code.encode(),
                    {"shape": list(payload.data.shape)},
                ),
            ),
        )
    )
    payload = ExecutableCurve(xr.DataArray(np.arange(5.0), dims="x"), "x + 1")
    tool = FigureComposerTool.from_source_payloads(
        {"curve": payload},
        sources=(FigureSourceState(name="curve", kind="test-executable"),),
        operations=(),
    )
    qtbot.addWidget(tool)
    saved = tool.to_dataset()
    expected = tool._code_trust_payload_entries()
    assert len(expected) == 1
    assert tool._code_trust_payload_entries_from_saved_dataset(saved) == expected
    import json

    manifest = json.loads(saved.attrs["figure_composer_source_manifest"])
    variable = manifest["sources"]["curve"]["arrays"]["curve"]
    incomplete = saved.drop_vars(variable)
    with pytest.raises(ValueError, match="Executable source data is missing"):
        tool._code_trust_payload_entries_from_saved_dataset(incomplete)


@pytest.mark.parametrize("array_name", [None, "signal", 42, ("signal", 2)])
def test_source_bundle_preserves_array_name_type(array_name):
    import json

    from erlab.interactive._figurecomposer._source_storage import (
        decode_source_bundle,
        encode_source_bundle,
    )

    data = xr.DataArray(np.arange(4.0), dims="x", name=array_name)
    bundle = encode_source_bundle({"curve": ArraySourcePayload(data)})
    # Manifest JSON must retain tuple and scalar names independently of slot names.
    manifest = json.loads(json.dumps(bundle.manifest))
    restored = decode_source_bundle(manifest, bundle.arrays)["curve"]
    assert isinstance(restored, ArraySourcePayload)
    assert type(restored.data.name) is type(array_name)
    xr.testing.assert_identical(restored.data, data)


def test_third_source_type_generated_code(curve_type, qtbot):
    import matplotlib.pyplot as plt

    tool = FigureComposerTool.from_source_payloads(
        {"curve": curve_type},
        sources=(FigureSourceState(name="curve", kind="test-curve"),),
        operations=(FigureOperationState.custom(label="Curve", code="ax.plot(curve)"),),
    )
    qtbot.addWidget(tool)
    code = tool.generated_code()
    namespace = {"curve": source_spec_for_kind("test-curve").safe_value(curve_type)}
    exec(code, namespace)  # noqa: S102
    figure = namespace["fig"]
    try:
        line = figure.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), np.arange(curve_type.data.size))
        np.testing.assert_array_equal(line.get_ydata(), curve_type.data.values)
        assert "_source_value" not in code
        assert "source_spec_for" not in code
    finally:
        plt.close(figure)


def test_third_source_type_native_clipboard(curve_type, qtbot):
    from erlab.interactive._figurecomposer import _rendering

    from ._common import _clear_clipboard, _select_operation_rows

    source = FigureComposerTool.from_source_payloads(
        {"curve": curve_type},
        sources=(FigureSourceState(name="curve", kind="test-curve"),),
        operations=(FigureOperationState.custom(label="Curve", code="ax.plot(curve)"),),
    )
    destination = FigureComposerTool.from_sources(
        {"curve": xr.DataArray(np.arange(3.0), dims="existing")},
        sources=(FigureSourceState(name="curve"),),
        operations=(),
    )
    qtbot.addWidget(source)
    qtbot.addWidget(destination)
    _clear_clipboard()
    _select_operation_rows(source, (0,))
    assert source._write_selected_operations_to_clipboard() == (0,)
    destination._paste_operations_from_clipboard()
    captured = destination.source_payload("curve_copy", _CurvePayload)
    assert captured is not None
    assert captured.unit == curve_type.unit
    xr.testing.assert_identical(captured.data, curve_type.data)
    assert destination.tool_status.operations[0].code == "ax.plot(curve_copy)"
    _rendering._render_into_figure(destination, destination.figure, sync_visible=False)
    assert not destination._operation_render_errors
    line = destination.figure.axes[0].lines[0]
    np.testing.assert_array_equal(line.get_ydata(), curve_type.data.values)
    assert set(destination.source_payloads()) == {"curve", "curve_copy"}


def test_mixed_source_panel_separates_type_shape_and_keeps_alias_first(qtbot):
    import lmfit

    from erlab.interactive._figurecomposer._fit_sources import FigureFitSourceSnapshot

    array = xr.DataArray(
        np.arange(7.0), dims="x", coords={"x": np.arange(7.0)}, name="measurement"
    )
    snapshot = FigureFitSourceSnapshot.from_dataset(
        array.xlm.modelfit("x", lmfit.models.LinearModel())
    )
    tool = FigureComposerTool.from_source_payloads(
        {"array": ArraySourcePayload(array), "fit": snapshot},
        sources=(
            FigureSourceState(name="array"),
            FigureSourceState(name="fit", kind="fit"),
        ),
        primary_source="array",
        operations=(),
    )
    qtbot.addWidget(tool)
    panel = tool.source_panel
    source_list = panel.source_list
    assert source_list.columnCount() == 3
    array_item = source_list.topLevelItem(0)
    fit_item = source_list.topLevelItem(1)
    assert array_item is not None
    assert fit_item is not None
    assert array_item.text(0) == "array"
    assert array_item.text(1) == source_spec_for_kind("array").label
    assert source_list.itemWidget(array_item, 1) is None
    assert array_item.text(2) == ""
    assert isinstance(source_list.itemWidget(array_item, 2), QtWidgets.QLabel)
    assert fit_item.text(0) == "fit"
    assert fit_item.text(1) == source_spec_for_kind("fit").label
    assert source_list.itemWidget(fit_item, 1) is None
    assert fit_item.text(2) == ""
    assert source_list.itemWidget(fit_item, 2) is None

    detail_layout = panel.source_detail_content.layout()
    assert detail_layout is not None
    assert detail_layout.itemAt(0).widget() is panel.source_alias_controls
    for name, metadata in (
        ("array", panel.source_inspector),
        ("fit", panel.source_editor_state_label),
    ):
        panel.set_selected_names((name,), current_name=name)
        tool._refresh_source_detail_panel()
        assert not panel.source_alias_controls.isHidden()
        assert panel.source_alias_edit.text() == name
        assert not metadata.isHidden()
        assert detail_layout.indexOf(
            panel.source_alias_controls
        ) < detail_layout.indexOf(metadata)

    panel.source_alias_edit.setText("renamed_fit")
    panel.source_alias_edit.editingFinished.emit()
    assert set(tool.source_payloads()) == {"array", "renamed_fit"}
    assert tool.source_payloads()["renamed_fit"] is snapshot
    assert panel.current_name() == "renamed_fit"
    assert panel.source_alias_edit.text() == "renamed_fit"
