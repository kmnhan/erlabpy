import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from pydantic import BaseModel, ValidationError
from qtpy import QtWidgets

import erlab
from erlab.interactive.derivative import DerivativeTool, dtool


@pytest.mark.parametrize("method_idx", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("interpmode", ["interp", "nointerp"])
@pytest.mark.parametrize(
    ("smoothmode", "nsmooth"),
    [
        ("none", 1),
        ("gaussian", 1),
        ("gaussian", 3),
        ("boxcar", 1),
        ("boxcar", 3),
    ],
)
def test_dtool(qtbot, interpmode, smoothmode, nsmooth, method_idx) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    def check_generated_code(w: DerivativeTool) -> None:
        namespace = {"era": erlab.analysis, "data": data, "np": np, "result": None}
        exec(w.copy_code(), {"__builtins__": {"range": range}}, namespace)  # noqa: S102
        xr.testing.assert_identical(w.result, namespace["result"])

    win.interp_group.setChecked(interpmode == "interp")
    win.smooth_group.setChecked(smoothmode != "none")
    win.sn_spin.setValue(nsmooth)

    match smoothmode:
        case "gaussian":
            win.smooth_combo.setCurrentIndex(0)
        case "boxcar":
            win.smooth_combo.setCurrentIndex(1)

    win.tab_widget.setCurrentIndex(method_idx)
    if method_idx == 1:
        win.lapl_factor_spin.setValue(40)
    elif method_idx == 3:
        win.curv_factor_spin.setValue(40)

    check_generated_code(win)
    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=True,
        state="stale",
    )

    # Test save & restore
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, DerivativeTool)

        assert win.tool_status == win_restored.tool_status
        assert str(win_restored.info_text) == str(win.info_text)
        assert win_restored.source_spec == win.source_spec
        assert win_restored.source_auto_update is True
        assert win_restored.source_state == "stale"
        check_generated_code(win_restored)


def test_dtool_update_data_preserves_state(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    new_data = xr.DataArray(
        np.arange(25, 50).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.interp_group.setChecked(True)
    win.nx_spin.setValue(7)
    win.ny_spin.setValue(9)
    win.smooth_group.setChecked(True)
    win.smooth_combo.setCurrentIndex(1)
    win.sx_spin.setValue(2)
    win.sy_spin.setValue(3)
    win.sn_spin.setValue(2)
    win.tab_widget.setCurrentIndex(3)
    win.curv_a0_spin.setValue(1.5)
    win.curv_factor_spin.setValue(12.0)

    status = win.tool_status
    win.update_data(new_data)

    assert win.tool_status == status
    xr.testing.assert_identical(win.tool_data, new_data)
    assert win.result.shape == win.processed_data.shape


def test_dtool_open_itool_uses_output_launcher(qtbot, monkeypatch) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    calls: list[tuple[xr.DataArray, str]] = []
    return_widget = QtWidgets.QWidget()
    qtbot.addWidget(return_widget)

    def _launch_stub(data: xr.DataArray, *, slot_key: str) -> QtWidgets.QWidget:
        calls.append((data.copy(deep=True), slot_key))
        return return_widget

    monkeypatch.setattr(win, "_launch_output_imagetool", _launch_stub)

    win.open_itool()

    assert calls
    assert calls[0][1] == "dtool.result"
    xr.testing.assert_identical(calls[0][0], win.result.T)
    assert win._itool is return_widget


def test_dtool_output_imagetool_provenance_transposes_result(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    xr.testing.assert_identical(win.output_imagetool_data("dtool.result"), win.result.T)

    spec = win.output_imagetool_provenance("dtool.result", win.result.T)

    assert spec is not None
    code = spec.display_code()
    assert code is not None
    assert "era.image.diffn(" in code
    assert code.endswith(".transpose()")


def test_dtool_source_update_marks_unavailable_for_incompatible_data(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.imagetool.provenance.selection(
            erlab.interactive.imagetool.provenance.TransposeOperation()
        ),
        auto_update=True,
    )

    parent_data = xr.DataArray(
        np.arange(125).reshape((5, 5, 5)),
        dims=("x", "y", "z"),
        coords={"x": np.arange(5), "y": np.arange(5), "z": np.arange(5)},
        name="data",
    )
    win.handle_parent_source_replaced(parent_data)

    assert win.source_state == "unavailable"
    xr.testing.assert_identical(win.tool_data, data)


def test_dtool_full_data_source_update_marks_unavailable_for_incompatible_data(
    qtbot,
) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=False,
    )

    parent_data = xr.DataArray(np.arange(5), dims=("x",), name="data")
    win.handle_parent_source_replaced(parent_data)

    assert win.source_state == "unavailable"
    xr.testing.assert_identical(win.tool_data, data)


def test_dtool_restored_source_binding_without_parent_stays_stale(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=True,
        state="stale",
    )

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, DerivativeTool)
        assert win_restored.source_state == "stale"

        assert win_restored._update_from_parent_source() is False
        assert win_restored.source_state == "stale"


def test_dtool_source_update_with_temporarily_missing_parent_stays_stale(qtbot) -> None:
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    updated = xr.DataArray(
        np.arange(25, 50).reshape((5, 5)), dims=["x", "y"], name="data"
    ).astype(np.float64)
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=True,
        state="stale",
    )

    available = False

    def fetcher() -> xr.DataArray:
        if not available:
            raise LookupError("Parent tool is temporarily unavailable")
        return updated

    win.set_source_parent_fetcher(fetcher)

    assert win._update_from_parent_source() is False
    assert win.source_state == "stale"

    available = True

    assert win._update_from_parent_source() is True
    assert win.source_state == "fresh"
    xr.testing.assert_identical(win.tool_data, updated)


def test_source_update_dialog_disables_auto_update_without_update_action(qtbot) -> None:
    dialog = erlab.interactive.utils._ToolSourceUpdateDialog(
        None, state="unavailable", auto_update=True
    )
    qtbot.addWidget(dialog)

    assert dialog.update_button.isEnabled() is False
    assert dialog.auto_update_check.isEnabled() is False


def test_tool_provenance_roundtrip_and_resolve_selection() -> None:
    encoded = erlab.interactive.imagetool.provenance.encode_provenance_value(
        {
            "outer": {
                "sel": slice(0.5, 2.5),
                "items": (slice(1, 4, 2), [np.int64(3)], {"beta": np.float64(1.5)}),
            }
        }
    )

    decoded = erlab.interactive.imagetool.provenance.decode_provenance_value(encoded)
    assert decoded == {
        "outer": {
            "sel": slice(0.5, 2.5),
            "items": (slice(1, 4, 2), [3], {"beta": 1.5}),
        }
    }

    parent = xr.DataArray(
        np.arange(24).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": [10.0, 11.0, 12.0, 13.0], "z": [5.0, 6.0]},
        name="data",
    )

    resolved_full = erlab.interactive.imagetool.provenance.full_data().apply(parent)
    xr.testing.assert_identical(resolved_full, parent)

    resolved_qsel = erlab.interactive.imagetool.provenance.selection(
        erlab.interactive.imagetool.provenance.QSelOperation(
            kwargs={"x": 1.0, "x_width": 1.0}
        )
    )
    xr.testing.assert_identical(
        resolved_qsel.apply(parent), parent.qsel(x=1.0, x_width=1.0)
    )

    resolved_selection = erlab.interactive.imagetool.provenance.selection(
        erlab.interactive.imagetool.provenance.IselOperation(
            kwargs={"x": slice(1, None), "z": 1}
        ),
        erlab.interactive.imagetool.provenance.SelOperation(
            kwargs={"y": slice(11.0, 12.0)}
        ),
        erlab.interactive.imagetool.provenance.SortCoordOrderOperation(),
        erlab.interactive.imagetool.provenance.TransposeOperation(dims=("y", "x")),
    )
    xr.testing.assert_identical(
        resolved_selection.apply(parent),
        parent.isel({"x": slice(1, None), "z": 1})
        .sel({"y": slice(11.0, 12.0)})
        .transpose("y", "x"),
    )

    resolved_squeezed = erlab.interactive.imagetool.provenance.selection(
        erlab.interactive.imagetool.provenance.IselOperation(kwargs={"z": 0}),
        erlab.interactive.imagetool.provenance.TransposeOperation(),
        erlab.interactive.imagetool.provenance.SqueezeOperation(),
    )
    xr.testing.assert_identical(
        resolved_squeezed.apply(parent),
        parent.isel({"z": 0}).transpose("y", "x").squeeze(),
    )

    parent_nonuniform_public = xr.DataArray(
        np.arange(24).reshape((4, 3, 2)),
        dims=("alpha", "eV", "beta"),
        coords={
            "alpha": [0.0, 0.6, 1.7, 3.0],
            "eV": [-0.2, 0.0, 0.2],
            "beta": [1.0, 2.0],
        },
        name="data",
    )
    parent_nonuniform = erlab.interactive.imagetool.slicer.make_dims_uniform(
        parent_nonuniform_public
    )
    resolved_nonuniform = erlab.interactive.imagetool.provenance.selection(
        erlab.interactive.imagetool.provenance.QSelOperation(kwargs={"beta": 2.0}),
        erlab.interactive.imagetool.provenance.IselOperation(
            kwargs={"alpha": slice(1, 3)}
        ),
        erlab.interactive.imagetool.provenance.SortCoordOrderOperation(),
    )
    xr.testing.assert_identical(
        resolved_nonuniform.apply(parent_nonuniform),
        parent_nonuniform_public.qsel(beta=2.0).isel({"alpha": slice(1, 3)}),
    )

    with pytest.raises(ValidationError, match="full_data', 'selection' or 'script"):
        erlab.interactive.imagetool.provenance.parse_tool_provenance_spec(
            {"kind": "invalid"}
        )

    with pytest.raises(ValidationError, match="Unknown provenance operation"):
        erlab.interactive.imagetool.provenance.parse_tool_provenance_spec(
            {"kind": "selection", "operations": [{"op": "invalid"}]}
        )


def test_tool_window_source_binding_helpers_and_failure_paths(qtbot) -> None:
    class _DummyState(BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data
            self._value = 0
            self.fail_validate = False
            self.fail_update = False

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState(value=self._value)

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            self._value = status.value

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def validate_update_data(self, new_data: xr.DataArray) -> xr.DataArray:
            if self.fail_validate:
                raise ValueError("invalid update")
            return super().validate_update_data(new_data)

        def update_data(self, new_data: xr.DataArray) -> None:
            if self.fail_update:
                raise RuntimeError("update failed")
            self._data = new_data

    data = xr.DataArray(np.arange(9).reshape((3, 3)), dims=("x", "y"), name="data")
    updated = xr.DataArray(
        np.arange(9, 18).reshape((3, 3)), dims=("x", "y"), name="data"
    )
    tool = _DummyTool(data)
    qtbot.addWidget(tool)

    original = QtWidgets.QLabel("original")
    replacement = QtWidgets.QLabel("replacement")
    tool.setCentralWidget(original)
    assert tool.centralWidget() is original
    tool.setCentralWidget(tool._tool_root_widget)
    tool._tool_content_widget = None
    assert tool.centralWidget() is tool._tool_root_widget
    tool._tool_content_widget = original
    tool.setCentralWidget(replacement)
    tool.setCentralWidget(replacement)
    assert tool.centralWidget() is replacement

    spec = erlab.interactive.imagetool.provenance.selection(
        erlab.interactive.imagetool.provenance.IselOperation(kwargs={"x": slice(0, 2)})
    )
    tool.set_source_binding(spec, auto_update=True, state="stale")
    assert tool.has_source_binding is True
    assert tool.source_status_text == "Source Update Available"
    assert "Automatic updates are enabled." in tool._source_status_button.toolTip()
    copied_spec = tool.source_spec
    assert copied_spec is not None
    with pytest.raises(ValidationError, match="Instance is frozen"):
        copied_spec.kind = "changed"
    assert tool.source_spec == spec

    tool._set_source_state("unavailable")
    assert tool.source_status_text == "Source Update Unavailable"
    assert "can no longer update" in tool._source_status_button.toolTip()

    tool.set_source_binding(None, auto_update=True, state="stale")
    assert tool.source_spec is None
    assert tool.has_source_binding is False
    assert tool.source_status_text == ""
    assert tool._source_status_bar.isHidden()

    with pytest.raises(RuntimeError, match="not bound to an ImageTool source"):
        tool._resolve_source_data(updated)

    tool._set_source_state("fresh")
    tool.handle_parent_source_replaced(updated)
    assert tool.source_state == "fresh"

    with pytest.raises(TypeError, match="ToolProvenanceSpec or None"):
        tool.set_source_binding(
            {"kind": "selection", "operations": [{"op": "invalid"}]}
        )
    tool.set_source_parent_fetcher(lambda: updated)
    assert tool._update_from_parent_source() is False
    assert tool.source_state == "unavailable"

    tool.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(), auto_update=True
    )
    tool.fail_validate = True
    assert tool._update_from_parent_source() is False
    assert tool.source_state == "unavailable"
    tool.fail_validate = False

    tool.fail_update = True
    assert tool._update_from_parent_source() is False
    assert tool.source_state == "unavailable"
    tool.fail_update = False

    assert tool._update_from_parent_source() is True
    assert tool.source_state == "fresh"
    xr.testing.assert_identical(tool.tool_data, updated)

    tool.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(), auto_update=False
    )
    tool.handle_parent_source_replaced(updated * 2)
    assert tool.source_state == "stale"

    tool.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(), auto_update=True
    )
    tool.fail_validate = True
    tool.handle_parent_source_replaced(updated)
    assert tool.source_state == "unavailable"
    tool.fail_validate = False

    tool.fail_update = True
    tool.handle_parent_source_replaced(updated)
    assert tool.source_state == "unavailable"
    tool.fail_update = False

    bad_parent = SimpleNamespace(
        kspace=SimpleNamespace(
            _valid_offset_keys=(),
            momentum_axes=(),
            configuration=0,
            _has_hv=False,
        )
    )
    tool.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(), auto_update=True
    )
    tool.handle_parent_source_replaced(bad_parent)
    assert tool.source_state == "unavailable"


def test_tool_copy_code_includes_parent_lineage_for_standalone_imagetool(qtbot) -> None:
    class _DummyState(BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState()

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def current_provenance_spec(
            self,
        ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
            return self._compose_with_input_provenance(
                lambda input_name: erlab.interactive.imagetool.provenance.script(
                    erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                        label="Compute dummy output",
                        code=f"result = {(input_name or 'data')}.mean()",
                    ),
                    start_label="Start from current dummy-tool input data",
                )
            )

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

    data = xr.DataArray(np.arange(9).reshape((3, 3)), dims=("x", "y"), name="data")
    parent = erlab.interactive.itool(data, execute=False, manager=False)
    assert isinstance(parent, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(parent)
    parent.set_provenance_spec(
        erlab.interactive.imagetool.provenance.selection(
            erlab.interactive.imagetool.provenance.IselOperation(
                kwargs={"x": slice(0, 2)}
            )
        )
    )

    tool = _DummyTool(data.isel(x=slice(0, 2)))
    qtbot.addWidget(tool)
    tool.set_source_binding(erlab.interactive.imagetool.provenance.full_data())
    parent.slicer_area.add_tool_window(tool, transfer_to_manager=False)

    code = tool.copy_code()

    assert code == "result = data.isel(x=slice(0, 2)).mean()"


def test_tool_input_provenance_snapshot_tracks_applied_refreshes(qtbot) -> None:
    class _DummyState(BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState()

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def current_provenance_spec(
            self,
        ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
            return self._compose_with_input_provenance(
                lambda input_name: erlab.interactive.imagetool.provenance.script(
                    erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                        label="Compute dummy output",
                        code=f"result = {(input_name or 'data')}.mean()",
                    ),
                    start_label="Start from current dummy-tool input data",
                )
            )

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

    data = xr.DataArray(np.arange(16).reshape((4, 4)), dims=("x", "y"), name="data")
    parent_provenance = {
        "spec": erlab.interactive.imagetool.provenance.selection(
            erlab.interactive.imagetool.provenance.IselOperation(
                kwargs={"x": slice(0, 2)}
            )
        )
    }

    tool = _DummyTool(data.isel(x=slice(0, 2)))
    qtbot.addWidget(tool)
    tool.set_source_binding(erlab.interactive.imagetool.provenance.full_data())
    tool.set_input_provenance_parent_fetcher(lambda: parent_provenance["spec"])

    initial_code = tool.copy_code()
    assert initial_code == "result = data.isel(x=slice(0, 2)).mean()"

    parent_provenance["spec"] = erlab.interactive.imagetool.provenance.selection(
        erlab.interactive.imagetool.provenance.IselOperation(kwargs={"y": slice(0, 2)})
    )
    stale_code = tool.copy_code()
    assert stale_code == "result = data.isel(x=slice(0, 2)).mean()"

    tool._data = data.isel(y=slice(0, 2))
    tool.finalize_source_refresh()

    refreshed_code = tool.copy_code()
    assert refreshed_code == "result = data.isel(y=slice(0, 2)).mean()"


def test_tool_input_provenance_resyncs_when_parent_fetcher_arrives_late(qtbot) -> None:
    class _DummyState(BaseModel):
        value: int = 0

    class _DummyTool(erlab.interactive.utils.ToolWindow[_DummyState]):
        StateModel = _DummyState
        tool_name = "dummy"

        def __init__(self, data: xr.DataArray) -> None:
            super().__init__()
            self._data = data

        @property
        def tool_status(self) -> _DummyState:
            return _DummyState()

        @tool_status.setter
        def tool_status(self, status: _DummyState) -> None:
            del status

        @property
        def tool_data(self) -> xr.DataArray:
            return self._data

        def current_provenance_spec(
            self,
        ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
            return self._compose_with_input_provenance(
                lambda input_name: erlab.interactive.imagetool.provenance.script(
                    erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                        label="Compute dummy output",
                        code=f"result = {(input_name or 'data')}.mean()",
                    ),
                    start_label="Start from current dummy-tool input data",
                )
            )

        def update_data(self, new_data: xr.DataArray) -> None:
            self._data = new_data

    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(16).reshape((4, 4)), dims=("x", "y"), name="data")

    tool = _DummyTool(data)
    qtbot.addWidget(tool)
    tool.set_source_binding(prov.selection(prov.SqueezeOperation()))
    tool.set_input_provenance_parent_fetcher(lambda: None)

    early_code = tool.copy_code()
    assert early_code == "result = data.squeeze().mean()"

    tool.set_source_parent_fetcher(lambda: data)

    refreshed_code = tool.copy_code()
    assert refreshed_code == "result = data.mean()"
