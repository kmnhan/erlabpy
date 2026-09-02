from __future__ import annotations

import importlib.metadata
import json
import sys
import typing
from types import SimpleNamespace

import lmfit
import numpy as np
import pydantic
import pytest
import xarray as xr
from qtpy import QtWidgets

import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._trust as workspace_trust
import erlab.interactive.utils as interactive_utils
from erlab.interactive import _saved_tools
from erlab.interactive._code_trust import (
    create_entry,
    create_manifest,
    document_trust_has_trusted_lineage,
    document_trust_is_trusted,
    execution_capability_allows,
    new_document_trust,
    trusted_location_document_trust,
    untrusted_document_trust,
)
from erlab.interactive._code_trust._api import _document_trust_after_save
from erlab.interactive._code_trust._application import (
    load_document_trust,
    load_imported_document_trust,
    reset_saved_code_trust,
    save_document_trust,
)
from erlab.interactive._code_trust._core import CodeTrustReason
from erlab.interactive._code_trust._payloads import store_code_payload_entries
from erlab.interactive._figurecomposer import FigureComposerTool, FigureSourceState
from erlab.interactive._figurecomposer import _rendering as figure_rendering
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationState,
    FigureRecipeState,
)
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ReplayStep,
    ScriptInput,
    full_data,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    ModelFitOperation,
    ScriptCodeOperation,
    _ModelFitParameterSpec,
)
from erlab.interactive.imagetool._provenance._trust import provenance_code_trust_entries
from erlab.interactive.imagetool.manager._widgets import (
    _TrustedProvenanceReplayCancelled,
)
from erlab.interactive.imagetool.manager._workspace._trust import (
    current_workspace_code_trust_manifest,
    workspace_code_trust_manifest,
)
from erlab.interactive.utils import ToolWindow
from tests.interactive.imagetool.manager.workspace._support import (
    _current_workspace_payload_attrs,
    _current_workspace_payload_path,
    add_source_childtool,
)

if typing.TYPE_CHECKING:
    import pathlib


def _trust_uses_saved_signature(trust) -> bool:
    return trust.reason == CodeTrustReason.SIGNATURE


class _TrustProbeState(pydantic.BaseModel):
    value: int = 0


class _TrustProbeTool(ToolWindow):
    StateModel = _TrustProbeState

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _TrustProbeState()

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _TrustProbeState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _TrustProbeState) -> None:
        self._status = status


def _workspace_manifest(
    code: str,
    *,
    workspace_id: str = "workspace",
    tool_identifier: str = _saved_tools.FIGURE_COMPOSER_TOOL_ID,
) -> dict:
    recipe = FigureRecipeState(
        operations=(FigureOperationState.custom(label="custom", code=code),)
    )
    attrs = workspace_format._workspace_manifest_attrs(
        {
            "tool_cls_qualname": tool_identifier,
            "tool_state": recipe.model_dump_json(),
        }
    )
    return {
        "workspace_id": workspace_id,
        "root_order": [],
        "nodes": [
            {
                "kind": "tool",
                "path": "figures/0",
                "payload_attrs": attrs,
            }
        ],
    }


def _workspace_manifest_from_attrs(
    attrs: dict[str, typing.Any], *, kind: str = "imagetool", path: str = "0"
) -> dict:
    return {
        "nodes": [
            {
                "kind": kind,
                "path": path,
                "payload_attrs": workspace_format._workspace_manifest_attrs(attrs),
            }
        ]
    }


def _loaded_workspace_trust(
    manager,
    workspace_path: pathlib.Path,
    manifest: dict,
    *,
    selected_paths: set[str] | None = None,
):
    return manager._workspace_controller._loaded_workspace_code_trust(
        workspace_path, manifest, selected_paths=selected_paths
    )


def _load_workspace(manager, workspace_path: pathlib.Path, *, replace: bool = True):
    return manager._workspace_controller.loading._load_workspace_file(
        workspace_path,
        replace=replace,
        associate=False,
        mark_dirty=not replace,
        select=False,
    )


@pytest.fixture
def external_workspace(tmp_path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    workspace_path = tmp_path / "external.itws"
    workspace_path.touch()
    monkeypatch.setattr(workspace_trust, "workspace_path_is_trusted", lambda _: False)
    return workspace_path


def _serialized_imagetool_state(
    *,
    path: str = "scan.h5",
    target: str = "os:remove",
) -> dict:
    return {
        "file_path": path,
        "load_func": [
            target,
            {},
            FileDataSelection(kind="dataarray").model_dump(mode="json"),
        ],
    }


def _restore_imagetool(
    qtbot,
    *,
    state: dict[str, typing.Any] | None = None,
    attrs: dict[str, typing.Any] | None = None,
    trust=None,
) -> ImageTool:
    source = ImageTool(xr.DataArray(np.ones((2, 2)), dims=("x", "y")))
    qtbot.addWidget(source)
    dataset = source.to_dataset()
    if state:
        saved_state = json.loads(dataset.attrs["itool_state"])
        saved_state.update(state)
        dataset.attrs["itool_state"] = json.dumps(saved_state)
    if attrs:
        dataset.attrs.update(attrs)
    kwargs = {} if trust is None else {"_code_trust": trust}
    restored = ImageTool.from_dataset(dataset, **kwargs)
    qtbot.addWidget(restored)
    return restored


def _model_fit_operation() -> ModelFitOperation:
    return ModelFitOperation(
        fit_dim="x",
        model="PolynomialModel",
        model_kwargs={"degree": 1},
        parameters={
            "c0": _ModelFitParameterSpec(value=0.0),
            "c1": _ModelFitParameterSpec(expr="2 * c0"),
        },
        method="leastsq",
        parameter="c0",
    )


def _model_fit_source_spec():
    return full_data(_model_fit_operation())


def _script_source(code: str = "derived = data + 1"):
    return full_data(ScriptCodeOperation(label="Stored code", code=code))


def test_workspace_code_trust_manifest_uses_metadata_and_not_workspace_id() -> None:
    first = workspace_code_trust_manifest(
        _workspace_manifest("ax.plot([1])", workspace_id="first")
    )
    same_code = workspace_code_trust_manifest(
        _workspace_manifest("ax.plot([1])", workspace_id="attacker-copy")
    )
    changed_code = workspace_code_trust_manifest(
        _workspace_manifest("ax.plot([2])", workspace_id="first")
    )

    assert first.canonical_bytes() == same_code.canonical_bytes()
    assert first.canonical_bytes() != changed_code.canonical_bytes()


def test_workspace_trust_adapter_rejects_invalid_metadata() -> None:
    assert not workspace_trust.workspace_path_is_trusted("workspace.txt")
    assert workspace_trust._decoded_json_mapping("{") is None
    assert workspace_trust._decoded_json_mapping("[]") is None

    with pytest.raises(TypeError, match="attribute entry is invalid"):
        workspace_trust._entry_attrs({"payload_attrs": [["only-one-item"]]})
    with pytest.raises(TypeError, match="identifier must be a string"):
        workspace_trust._tool_code_trust_from_attrs({"tool_cls_qualname": 1})
    with pytest.raises(TypeError, match="state must be JSON text"):
        workspace_trust._tool_code_trust_from_attrs(
            {"tool_cls_qualname": "test:Tool", "tool_state": 1}
        )


def test_workspace_trust_adapter_filters_saved_attributes() -> None:
    assert workspace_trust._entry_attrs(
        {
            "payload_attrs": {
                "tool_state": "{}",
                "ignored": "not security metadata",
            }
        }
    ) == {"tool_state": "{}"}
    assert (
        workspace_trust._entry_attrs(
            {
                "payload_attrs": [
                    [
                        {"kind": "int", "value": 1},
                        {"kind": "str", "value": "ignored"},
                    ]
                ]
            }
        )
        == {}
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "erlab.interactive._figurecomposer:FigureComposerTool",
        "attacker.module:ToolWindow",
    ],
)
def test_saved_tool_class_registry_rejects_file_selected_imports(
    identifier: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = xr.Dataset(attrs={"tool_cls_qualname": identifier})

    def fail_if_imported(_module_name: str):
        raise AssertionError("document text selected a module import")

    monkeypatch.setattr(_saved_tools.importlib, "import_module", fail_if_imported)

    with pytest.raises(TypeError, match="not registered"):
        ToolWindow._saved_tool_class_from_dataset(dataset)
    with pytest.raises(TypeError, match="could not be inspected"):
        workspace_code_trust_manifest(
            _workspace_manifest(
                "run_user_code()",
                workspace_id="untrusted",
                tool_identifier=identifier,
            )
        )


def test_saved_tool_class_registry_accepts_canonical_figure_id() -> None:
    dataset = xr.Dataset(
        attrs={"tool_cls_qualname": _saved_tools.FIGURE_COMPOSER_TOOL_ID}
    )

    assert ToolWindow._saved_tool_class_from_dataset(dataset) is FigureComposerTool


def test_saved_tool_builtin_loader_resolves_nested_attributes() -> None:
    assert _saved_tools._load_builtin_tool("json", "JSONDecoder") is json.JSONDecoder


def test_saved_tool_class_registry_discovers_installed_extensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    module_name = "installed_erlab_tool_extension"
    identifier = f"{module_name}:InstalledTool"
    (tmp_path / f"{module_name}.py").write_text(
        "from erlab.interactive.utils import ToolWindow\n"
        "class InstalledTool(ToolWindow):\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    entry_point = importlib.metadata.EntryPoint(
        name="installed-tool",
        value=identifier,
        group="erlab.interactive.tool_windows",
    )
    monkeypatch.setattr(
        _saved_tools.importlib.metadata,
        "entry_points",
        lambda *, group: [entry_point],
    )
    monkeypatch.setattr(_saved_tools, "_ENTRY_POINTS_DISCOVERED", False)

    dataset = xr.Dataset(attrs={"tool_cls_qualname": identifier})
    restored_class = ToolWindow._saved_tool_class_from_dataset(dataset)

    assert restored_class.__module__ == module_name
    assert restored_class.__qualname__ == "InstalledTool"


def test_saved_tool_entry_point_does_not_replace_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[None] = []
    entry_point = SimpleNamespace(
        module="erlab.interactive._figurecomposer._tool",
        attr="FigureComposerTool",
        load=lambda: loaded.append(None),
    )
    monkeypatch.setattr(
        _saved_tools.importlib.metadata,
        "entry_points",
        lambda *, group: [entry_point],
    )
    monkeypatch.setattr(_saved_tools, "_ENTRY_POINTS_DISCOVERED", False)

    assert (
        _saved_tools.resolve_saved_tool_class(_saved_tools.FIGURE_COMPOSER_TOOL_ID)
        is FigureComposerTool
    )
    assert loaded == []


def test_saved_tool_registry_ignores_module_only_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _saved_tools.importlib.metadata,
        "entry_points",
        lambda *, group: [SimpleNamespace(attr=None)],
    )
    monkeypatch.setattr(_saved_tools, "_ENTRY_POINTS_DISCOVERED", False)

    _saved_tools._discover_saved_tool_entry_points()


def test_saved_tool_loader_must_return_a_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "test.saved_tools:NotAClass"
    monkeypatch.setattr(
        _saved_tools,
        "_SAVED_TOOL_CLASSES",
        dict(_saved_tools._SAVED_TOOL_CLASSES),
    )
    monkeypatch.setattr(
        _saved_tools,
        "_SAVED_TOOL_LOADERS",
        dict(_saved_tools._SAVED_TOOL_LOADERS),
    )
    monkeypatch.setattr(_saved_tools, "_ENTRY_POINTS_DISCOVERED", True)
    _saved_tools._register_saved_tool_loader(identifier, object)

    with pytest.raises(TypeError, match="is not a class"):
        _saved_tools.resolve_saved_tool_class(identifier)


def test_saved_tool_entry_point_must_load_its_declared_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "installed_erlab_tool_extension:DeclaredTool"
    entry_point = SimpleNamespace(
        module="installed_erlab_tool_extension",
        attr="DeclaredTool",
        load=lambda: FigureComposerTool,
    )
    monkeypatch.setattr(_saved_tools, "_SAVED_TOOL_CLASSES", {})
    monkeypatch.setattr(
        _saved_tools,
        "_SAVED_TOOL_LOADERS",
        dict(_saved_tools._SAVED_TOOL_LOADERS),
    )
    monkeypatch.setattr(
        _saved_tools.importlib.metadata,
        "entry_points",
        lambda *, group: [entry_point],
    )
    monkeypatch.setattr(_saved_tools, "_ENTRY_POINTS_DISCOVERED", False)

    with pytest.raises(TypeError, match=r"loaded .*FigureComposerTool"):
        _saved_tools.resolve_saved_tool_class(identifier)


def test_workspace_uses_the_registered_tool_trust_hook() -> None:
    class FutureState(pydantic.BaseModel):
        code: str

    class FutureTool(ToolWindow):
        StateModel = FutureState
        _CODE_TRUST_DOMAIN = "test.future-tool"
        _CODE_TRUST_POLICY_VERSION = 7

        @classmethod
        def _code_trust_entries_from_status(cls, status: FutureState):
            return (
                create_entry(
                    "test.future-code",
                    "tool/code",
                    status.code,
                ),
            )

    manifest = workspace_code_trust_manifest(
        _workspace_manifest_from_attrs(
            {
                "tool_cls_qualname": FutureTool._qual_name(),
                "tool_state": FutureState(code="run_user_code()").model_dump_json(),
            },
            kind="tool",
            path="tools/0",
        )
    )

    assert [entry.feature for entry in manifest.entries] == ["test.future-code"]
    assert manifest.entries[0].location == "tools/0/tool/code"
    assert manifest.entries[0].context == {}


def test_workspace_manifest_includes_serialized_fit_callables(qtbot) -> None:
    from erlab.interactive._fit1d import Fit1DTool

    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    model = lmfit.Model(lambda x, slope=1.0, intercept=0.0: slope * x + intercept)
    params = model.make_params()
    params["intercept"].expr = "2 * slope"
    tool = Fit1DTool(data, model=model, params=params)
    qtbot.addWidget(tool)
    tool._serialized_fit_result_blob = np.arange(8, dtype=np.uint8)
    tool._pending_persisted_fit_is_current = False
    saved = tool.to_dataset()
    manifest = workspace_code_trust_manifest(
        _workspace_manifest_from_attrs(saved.attrs, kind="tool", path="tools/0")
    )

    features = [entry.feature for entry in manifest.entries]
    assert features[0] == "erlab.fit.serialized-model"
    assert "erlab.fit.parameter-expression" in features
    assert features[-1] == "erlab.fit.serialized-result"


def test_signed_workspace_rejects_payload_metadata_added_outside_manifest(
    qtbot, monkeypatch, tmp_path, manager_context
) -> None:
    from erlab.interactive._fit1d import Fit1DTool

    reset_saved_code_trust(domain="erlab.workspace")
    workspace_path = tmp_path / "injected-pending-payload.itws"
    data = xr.DataArray(np.arange(5.0), dims="x", name="data")
    monkeypatch.setattr(
        QtWidgets.QDialog,
        "exec",
        lambda dialog: pytest.fail(
            f"Unexpected dialog: {type(dialog).__name__} {dialog.windowTitle()!r}"
        ),
    )

    with manager_context() as manager:
        manager.add_imagetool(ImageTool(data), show=False)
        fit_tool = Fit1DTool(data, model=lmfit.Model(lambda x: x))
        qtbot.addWidget(fit_tool)
        fit_uid = add_source_childtool(manager, fit_tool, 0, show=False)
        fit_tool.hide()

        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        assert _trust_uses_saved_signature(manager._workspace_state.code_trust)

    injected_blob = np.arange(8, dtype=np.uint8)
    injected_attrs: dict[str, typing.Any] = {}
    store_code_payload_entries(
        injected_attrs,
        (Fit1DTool._fit_result_code_trust_entry(injected_blob),),
    )
    xr.Dataset(
        {
            Fit1DTool._PERSISTED_FIT_RESULT_VAR: xr.DataArray(
                injected_blob,
                dims=(Fit1DTool._PERSISTED_FIT_RESULT_DIM,),
            )
        },
        attrs=injected_attrs,
    ).to_netcdf(
        workspace_path,
        mode="a",
        group=_current_workspace_payload_path(
            workspace_path, f"0/childtools/{fit_uid}"
        ).lstrip("/"),
        engine="h5netcdf",
    )

    decode_calls: list[None] = []

    def fail_if_decoded(_payload):
        decode_calls.append(None)
        raise AssertionError("unsigned pending payload was decoded")

    monkeypatch.setattr(
        interactive_utils,
        "_deserialize_fit_dataset_blob",
        fail_if_decoded,
    )

    with manager_context() as manager:
        assert _load_workspace(manager, workspace_path)
        assert _trust_uses_saved_signature(manager._workspace_state.code_trust)
        node = manager._child_node(fit_uid)
        assert node.pending_workspace_tool_payload is not None
        assert node.materialize_pending_workspace_payload()

        assert decode_calls == []
        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert node.tool_window is not None
        assert not document_trust_has_trusted_lineage(node.tool_window._document_trust)


def test_signed_workspace_materializes_unchanged_fit_payload(
    qtbot, tmp_path, manager_context
) -> None:
    from erlab.interactive._fit1d import Fit1DTool

    reset_saved_code_trust(domain="erlab.workspace")
    workspace_path = tmp_path / "signed-fit-payload.itws"
    data = xr.DataArray(np.arange(5.0), dims="x", name="data")

    with manager_context() as manager:
        manager.add_imagetool(ImageTool(data), show=False)
        fit_tool = Fit1DTool(data)
        qtbot.addWidget(fit_tool)
        fit_tool._serialized_fit_result_blob = np.arange(8, dtype=np.uint8)
        fit_tool._pending_persisted_fit_is_current = False
        fit_uid = add_source_childtool(manager, fit_tool, 0, show=False)
        fit_tool.hide()

        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        assert _trust_uses_saved_signature(manager._workspace_state.code_trust)

    with manager_context() as manager:
        assert _load_workspace(manager, workspace_path)
        signed = manager._workspace_state.code_trust
        assert _trust_uses_saved_signature(signed)
        node = manager._child_node(fit_uid)
        assert node.pending_workspace_tool_payload is not None

        assert node.materialize_pending_workspace_payload()

        assert manager._workspace_state.code_trust == signed
        assert node.tool_window is not None
        assert node.tool_window._document_trust == signed


def test_signed_workspace_materializes_child_parent_source_code(
    qtbot, tmp_path, manager_context
) -> None:
    reset_saved_code_trust(domain="erlab.workspace")
    workspace_path = tmp_path / "signed-child-parent-source.itws"
    parent_data = xr.DataArray(
        np.arange(10.0).reshape(2, 5),
        dims=("y", "x"),
        name="parent",
    )
    child_data = xr.DataArray(np.arange(2.0), dims="y", name="child")
    source_spec = _model_fit_source_spec()

    with manager_context() as manager:
        manager.add_imagetool(ImageTool(parent_data), show=False)
        child = _TrustProbeTool(child_data)
        qtbot.addWidget(child)
        child.set_script_inputs(
            (
                ScriptInput(
                    name="data",
                    data_role="displayed",
                    source_spec=source_spec.model_dump(mode="json"),
                ),
            ),
            primary_input="data",
        )
        parent_uid = manager._node_for_target(0).uid
        child_uid = manager.add_childtool(
            child,
            script_inputs={"data": 0},
            show=True,
        )

        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        references = json.loads(
            _current_workspace_payload_attrs(
                workspace_path, f"0/childtools/{child_uid}"
            )["tool_data_references"]
        )
        reference = references[interactive_utils._SAVED_TOOL_DATA_NAME]
        assert reference["kind"] == "manager_node"
        assert reference["input_name"] == "data"
        assert reference["node_uid"] == parent_uid
        assert reference["data_role"] == "displayed"
        assert reference["source_spec"] == source_spec.model_dump(mode="json")
        assert _trust_uses_saved_signature(manager._workspace_state.code_trust)

    with manager_context() as manager:
        assert _load_workspace(manager, workspace_path)
        signed = manager._workspace_state.code_trust
        assert _trust_uses_saved_signature(signed)
        node = manager._child_node(child_uid)

        assert manager._workspace_state.code_trust == signed
        assert node.tool_window is not None
        assert node.tool_window._document_trust == signed
        assert node.tool_window.script_inputs[0].parsed_source_spec() == source_spec
        assert node.tool_window.tool_data.dims == ("y",)


def test_workspace_code_trust_manifest_limits_selected_import_paths() -> None:
    manifest = _workspace_manifest("ax.plot([1])")

    selected = workspace_code_trust_manifest(manifest, selected_paths={"figures/0"})
    excluded = workspace_code_trust_manifest(manifest, selected_paths={"other"})

    assert selected.has_executable_code
    assert not excluded.has_executable_code


@pytest.mark.parametrize(
    ("attribute", "location_segment"),
    [
        ("manager_node_live_source_spec", "/source/provenance/"),
        ("tool_source_spec", "/tool-source/provenance/"),
    ],
)
def test_workspace_manifest_includes_saved_live_source_code(
    attribute: str,
    location_segment: str,
) -> None:
    workspace_manifest = _workspace_manifest_from_attrs(
        {attribute: _model_fit_source_spec().model_dump_json()},
        kind="imagetool" if attribute.startswith("manager_") else "tool",
        path="selected",
    )

    selected = workspace_code_trust_manifest(
        workspace_manifest,
        selected_paths={"selected"},
    )
    excluded = workspace_code_trust_manifest(
        workspace_manifest,
        selected_paths={"other"},
    )

    assert [entry.code for entry in selected.entries] == ["2 * c0"]
    assert location_segment in selected.entries[0].location
    assert not excluded.has_executable_code


def test_imported_deferred_source_code_makes_combined_workspace_untrusted(
    external_workspace,
    manager_context,
) -> None:
    workspace_manifest = _workspace_manifest_from_attrs(
        {
            "manager_node_live_source_spec": (
                _model_fit_source_spec().model_dump_json()
            ),
            "manager_node_source_state": "stale",
        },
        path="selected",
    )

    with manager_context() as manager:
        controller = manager._workspace_controller
        incoming = _loaded_workspace_trust(
            manager,
            external_workspace,
            workspace_manifest,
            selected_paths={"selected"},
        )
        assert not document_trust_is_trusted(incoming)
        assert document_trust_is_trusted(manager._workspace_state.code_trust)

        assert controller._load_with_code_trust(
            incoming, replace=False, load=lambda: True
        )

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)


def test_workspace_trust_revocation_notifies_every_tool(
    manager_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def failing_callback() -> None:
        calls.append("first")
        raise RuntimeError("failed callback")

    with manager_context() as manager, monkeypatch.context() as patcher:
        patcher.setattr(
            manager._tool_graph,
            "nodes",
            {
                "first": SimpleNamespace(
                    uid="first",
                    tool_window=SimpleNamespace(_code_trust_changed=failing_callback),
                ),
                "second": SimpleNamespace(
                    uid="second",
                    tool_window=SimpleNamespace(
                        _code_trust_changed=lambda: calls.append("second")
                    ),
                ),
            },
        )

        manager._workspace_controller._set_workspace_code_trust(
            untrusted_document_trust()
        )

    assert calls == ["first", "second"]


def test_failed_workspace_load_restores_trust_notification(
    manager_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    with manager_context() as manager:
        controller = manager._workspace_controller
        notifications: list[None] = []
        monkeypatch.setattr(
            controller,
            "_notify_code_trust_changed",
            lambda: notifications.append(None),
        )

        assert not controller._load_with_code_trust(
            untrusted_document_trust(), replace=True, load=lambda: False
        )

        assert document_trust_is_trusted(manager._workspace_state.code_trust)
        assert len(notifications) == 2


def test_workspace_load_binds_trust_manifest_once_after_node_batch(
    manager_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = create_manifest(
        workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN,
        workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION,
        (create_entry("test.code", "nodes/source", "run_code()"),),
    )
    manifest_calls: list[None] = []

    with manager_context() as manager:
        controller = manager._workspace_controller
        monkeypatch.setattr(controller, "_mark_workspace_dirty", lambda **_: False)
        monkeypatch.setattr(
            workspace_trust,
            "current_workspace_code_trust_manifest",
            lambda _manager: manifest_calls.append(None) or manifest,
        )

        def load() -> bool:
            for uid in ("first", "second", "third"):
                controller._mark_node_added(uid)
            assert not manifest_calls
            return True

        assert controller._load_with_code_trust(
            untrusted_document_trust(), replace=True, load=load
        )

        assert manifest_calls == [None]
        assert controller._code_trust_manifest_binding_depth == 0
        assert not controller._code_trust_manifest_binding_pending


@pytest.mark.parametrize("raises", [False, True])
def test_failed_workspace_load_discards_pending_manifest_binding(
    manager_context, monkeypatch: pytest.MonkeyPatch, *, raises: bool
) -> None:
    manifest_calls: list[None] = []

    with manager_context() as manager:
        controller = manager._workspace_controller
        previous = manager._workspace_state.code_trust
        monkeypatch.setattr(
            workspace_trust,
            "current_workspace_code_trust_manifest",
            lambda _manager: manifest_calls.append(None),
        )

        def load() -> bool:
            controller._bind_current_workspace_manifest_if_review_needed()
            if raises:
                raise RuntimeError("load failed")
            return False

        if raises:
            with pytest.raises(RuntimeError, match="load failed"):
                controller._load_with_code_trust(
                    untrusted_document_trust(), replace=True, load=load
                )
        else:
            assert not controller._load_with_code_trust(
                untrusted_document_trust(), replace=True, load=load
            )

        assert manager._workspace_state.code_trust == previous
        assert not manifest_calls
        assert controller._code_trust_manifest_binding_depth == 0
        assert not controller._code_trust_manifest_binding_pending


def test_workspace_load_restores_trust_if_deferred_manifest_binding_fails(
    manager_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    with manager_context() as manager:
        controller = manager._workspace_controller
        previous = manager._workspace_state.code_trust

        def fail_manifest(_manager):
            raise RuntimeError("manifest failed")

        monkeypatch.setattr(
            workspace_trust,
            "current_workspace_code_trust_manifest",
            fail_manifest,
        )

        def load() -> bool:
            controller._bind_current_workspace_manifest_if_review_needed()
            return True

        with pytest.raises(RuntimeError, match="manifest failed"):
            controller._load_with_code_trust(
                untrusted_document_trust(), replace=True, load=load
            )

        assert manager._workspace_state.code_trust == previous
        assert controller._code_trust_manifest_binding_depth == 0
        assert not controller._code_trust_manifest_binding_pending


def test_selected_workspace_import_retains_complete_manifest_signature(
    external_workspace, manager_context
) -> None:
    reset_saved_code_trust(domain=workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN)
    manifest = _workspace_manifest(
        "first_selected_import_marker()",
        workspace_id="signed-import",
    )
    second = _workspace_manifest(
        "second_selected_import_marker()",
        workspace_id="signed-import",
    )["nodes"][0]
    second["path"] = "figures/1"
    manifest["nodes"].append(second)
    complete = workspace_code_trust_manifest(manifest)
    selected = workspace_code_trust_manifest(
        manifest,
        selected_paths={"figures/0"},
    )
    assert complete.canonical_bytes() != selected.canonical_bytes()
    _saved, signature_stored = save_document_trust(
        new_document_trust(),
        complete,
    )
    assert signature_stored

    try:
        with manager_context() as manager:
            imported = _loaded_workspace_trust(
                manager,
                external_workspace,
                manifest,
                selected_paths={"figures/0"},
            )
        assert _trust_uses_saved_signature(imported)
        forged = create_manifest(
            workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN,
            workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION,
            (
                create_entry(
                    "test.forged-import",
                    "figures/0/forged",
                    "forged_import_marker()",
                ),
            ),
        )
        assert not _trust_uses_saved_signature(
            load_imported_document_trust(complete, forged)
        )
    finally:
        reset_saved_code_trust(domain=workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN)


def test_signed_nonreplacing_import_into_untrusted_workspace_fails_closed_on_collision(
    manager_context,
) -> None:
    existing_entry = create_entry(
        "test.code",
        "figures/n0/figure/operations/0",
        "same_code()",
    )
    manifest = create_manifest(
        workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN,
        workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION,
        (existing_entry,),
    )
    signed = _document_trust_after_save(
        new_document_trust(),
        manifest,
        saved_trusted_lineage=True,
        signature_stored=True,
    )

    with manager_context() as manager:
        manager._workspace_state.code_trust = untrusted_document_trust(manifest)

        assert manager._workspace_controller._load_with_code_trust(
            signed,
            replace=False,
            load=lambda: True,
        )

        trust = manager._workspace_state.code_trust
        assert not document_trust_is_trusted(trust)
        assert not manager._workspace_controller.issue_code_execution_capability(
            (existing_entry,)
        )


def test_selected_workspace_import_ignores_unavailable_unselected_tool(
    external_workspace, manager_context
) -> None:
    manifest = _workspace_manifest(
        "selected_import_marker()",
        workspace_id="partial-import",
    )
    unavailable = _workspace_manifest(
        "unavailable_import_marker()",
        workspace_id="partial-import",
        tool_identifier="unavailable.extension:Tool",
    )["nodes"][0]
    unavailable["path"] = "figures/1"
    manifest["nodes"].append(unavailable)
    with manager_context() as manager:
        imported = _loaded_workspace_trust(
            manager,
            external_workspace,
            manifest,
            selected_paths={"figures/0"},
        )

    assert not document_trust_is_trusted(imported)


@pytest.mark.parametrize("selected_paths", [None, {"figures/0"}])
def test_workspace_with_unavailable_selected_tool_loads_untrusted(
    selected_paths, external_workspace, manager_context
) -> None:
    manifest = _workspace_manifest(
        "unavailable_import_marker()",
        workspace_id="unavailable-tool",
        tool_identifier="unavailable.extension:Tool",
    )
    with manager_context() as manager:
        loaded_trust = _loaded_workspace_trust(
            manager,
            external_workspace,
            manifest,
            selected_paths=selected_paths,
        )

    assert not document_trust_is_trusted(loaded_trust)


def test_workspace_tool_extension_failure_loads_untrusted(
    external_workspace, monkeypatch, manager_context
) -> None:
    manifest = _workspace_manifest(
        "extension_marker()",
        workspace_id="broken-extension",
    )
    monkeypatch.setattr(
        workspace_trust,
        "resolve_saved_tool_class",
        lambda _identifier: (_ for _ in ()).throw(RuntimeError("extension failed")),
    )

    with manager_context() as manager:
        loaded_trust = _loaded_workspace_trust(manager, external_workspace, manifest)

    assert not document_trust_is_trusted(loaded_trust)


@pytest.mark.parametrize("current_document", [True, False])
def test_uninspectable_saved_workspace_fails_closed_for_current_document(
    current_document: bool, monkeypatch, manager_context
) -> None:
    save_calls: list[None] = []
    monkeypatch.setattr(
        workspace_trust,
        "workspace_code_trust_manifest",
        lambda _manifest: (_ for _ in ()).throw(TypeError("unavailable tool")),
    )
    monkeypatch.setattr(
        "erlab.interactive.imagetool.manager._workspace._controller.save_document_trust",
        lambda *_args, **_kwargs: save_calls.append(None),
    )

    with manager_context() as manager:
        initial_trust = manager._workspace_state.code_trust
        manager._workspace_controller._record_saved_workspace_code_trust(
            {},
            trusted_lineage=True,
            current_document=current_document,
        )

        assert save_calls == []
        if current_document:
            assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        else:
            assert manager._workspace_state.code_trust is initial_trust


def test_workspace_review_without_remaining_code_clears_warning(
    monkeypatch, manager_context
) -> None:
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected dialog")),
    )
    with manager_context() as manager:
        manager._workspace_state.code_trust = untrusted_document_trust()
        manager._workspace_controller._refresh_code_trust_ui()

        manager._workspace_controller.review_and_approve_workspace_code_trust()

        assert document_trust_is_trusted(manager._workspace_state.code_trust)


@pytest.mark.parametrize(
    "code",
    [
        "derived = data + 1",
        "import os\nderived = data + int(os.path.exists('/'))",
        (
            'builtins_dict = vars(erlab)["__builtins__"]\n'
            'os = builtins_dict["__import__"]("os")\n'
            "derived = data + int(os.path.exists(os.devnull))"
        ),
    ],
)
@pytest.mark.parametrize(
    "provenance_attr",
    ["manager_node_provenance_spec", "itool_provenance_spec"],
)
def test_workspace_code_trust_manifest_includes_live_python_provenance(
    code: str,
    provenance_attr: str,
) -> None:
    provenance = _script_source(code)
    manifest = workspace_code_trust_manifest(
        _workspace_manifest_from_attrs(
            {provenance_attr: provenance.model_dump_json()}, kind="imagetool"
        )
    )

    assert manifest.has_executable_code
    assert manifest.entries[0].feature == "erlab.provenance.script-code"


def test_workspace_manifest_deduplicates_saved_imagetool_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = _script_source()
    serialized = provenance.model_dump_json()
    calls: list[None] = []

    def counted_entries(spec, *, location_prefix):
        calls.append(None)
        return provenance_code_trust_entries(spec, location_prefix=location_prefix)

    monkeypatch.setattr(
        workspace_trust,
        "provenance_code_trust_entries",
        counted_entries,
    )
    manifest = workspace_code_trust_manifest(
        _workspace_manifest_from_attrs(
            {
                "itool_provenance_spec": serialized,
                "manager_node_provenance_spec": serialized,
            }
        )
    )

    assert len(manifest.entries) == 1
    assert calls == [None]


def test_workspace_manifest_ignores_legacy_tool_manager_provenance() -> None:
    recipe = FigureRecipeState(
        operations=(FigureOperationState.custom(label="custom", code="ax.plot([1])"),)
    )
    manifest = workspace_code_trust_manifest(
        _workspace_manifest_from_attrs(
            {
                "tool_cls_qualname": _saved_tools.FIGURE_COMPOSER_TOOL_ID,
                "tool_state": recipe.model_dump_json(),
                "manager_node_provenance_spec": _script_source().model_dump_json(),
            },
            kind="tool",
            path="figures/0",
        )
    )

    assert [entry.feature for entry in manifest.entries] == [
        "erlab.figure-composer.custom-code"
    ]
    assert [entry.code for entry in manifest.entries] == ["ax.plot([1])"]


def test_current_workspace_manifest_ignores_tool_displayed_provenance(
    qtbot, manager_context
) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            operations=(
                FigureOperationState.custom(label="custom", code="ax.plot([1])"),
            )
        ),
    )
    qtbot.addWidget(tool)

    with manager_context() as manager:
        uid = manager.add_figuretool(tool, show=False)
        manager._tool_graph.nodes[uid].set_pending_workspace_payload(
            "tool",
            "pending.itws",
            "nodes/figures/0/payload",
            payload_attrs={
                "manager_node_provenance_spec": _script_source().model_dump_json()
            },
        )

        manifest = current_workspace_code_trust_manifest(manager)

    assert [entry.feature for entry in manifest.entries] == [
        "erlab.figure-composer.custom-code"
    ]
    assert [entry.code for entry in manifest.entries] == ["ax.plot([1])"]


def test_current_workspace_manifest_includes_distinct_pending_itool_provenance(
    qtbot,
    tmp_path,
    manager_context,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    tool = ImageTool(data)
    qtbot.addWidget(tool)
    safe = full_data(AverageOperation(dims=("x",)))
    executable = _script_source()
    attrs = {
        "manager_node_provenance_spec": safe.model_dump_json(),
        "itool_provenance_spec": executable.model_dump_json(),
    }

    with manager_context() as manager:
        manager.add_imagetool(tool, show=False)
        node = manager._node_for_target(0)
        node.set_pending_workspace_payload(
            "imagetool",
            tmp_path / "pending.itws",
            "nodes/0/payload",
            payload_attrs=attrs,
        )

        current = current_workspace_code_trust_manifest(manager)

    loaded = workspace_code_trust_manifest(_workspace_manifest_from_attrs(attrs))
    assert current.entries == loaded.entries
    assert [entry.code for entry in current.entries] == ["derived = data + 1"]


def test_workspace_code_trust_manifest_includes_model_fit_expressions() -> None:
    provenance = _model_fit_source_spec()
    manifest = workspace_code_trust_manifest(
        _workspace_manifest_from_attrs(
            {"manager_node_provenance_spec": provenance.model_dump_json()}
        )
    )

    assert [entry.feature for entry in manifest.entries] == [
        "erlab.provenance.model-fit-parameter-expression"
    ]
    assert manifest.entries[0].code == "2 * c0"


def test_manager_provenance_apply_authorizes_model_fit_expression_before_execution(
    manager_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), coords={"x": np.arange(4.0)})
    operation = _model_fit_operation()
    provenance = full_data(operation)
    executions: list[None] = []
    monkeypatch.setattr(
        ModelFitOperation,
        "apply",
        lambda _operation, source: executions.append(None) or source,
    )

    with manager_context() as manager:
        manager._workspace_state.code_trust = untrusted_document_trust()
        with pytest.raises(_TrustedProvenanceReplayCancelled):
            manager._apply_provenance(
                provenance,
                data,
                reason="test provenance authorization",
            )
        assert executions == []

        manager._workspace_state.code_trust = new_document_trust()
        result = manager._apply_provenance(
            provenance,
            data,
            reason="test provenance authorization",
        )

    xr.testing.assert_identical(result, data)
    assert executions == [None]


def test_provenance_manifest_ignores_safe_pipeline_and_review_state() -> None:
    def manifest(
        *,
        average_dim: str = "x",
        start_label: str = "Calculate result",
        code_label: str = "Run code",
        input_label: str = "Displayed input",
        node_uid: str = "node-a",
        snapshot_id: str = "snapshot-a",
    ) -> bytes:
        provenance = script(
            ScriptCodeOperation(
                label=code_label,
                code=("import os\nderived = data + int(os.path.exists(os.devnull))"),
            ),
            start_label=start_label,
            active_name="derived",
            steps=(
                ReplayStep(
                    operation=AverageOperation(dims=(average_dim,)),
                    input_policy="current",
                ),
            ),
            script_inputs=(
                ScriptInput(
                    name="data",
                    label=input_label,
                    node_uid=node_uid,
                    node_snapshot_token=snapshot_id,
                    provenance_spec=full_data().model_dump(mode="json"),
                ),
            ),
        )
        return create_manifest(
            "test.provenance",
            1,
            provenance_code_trust_entries(
                provenance,
                location_prefix="provenance",
            ),
        ).canonical_bytes()

    baseline = manifest()

    assert baseline == manifest(average_dim="y")
    assert baseline == manifest(node_uid="node-b")
    assert baseline == manifest(snapshot_id="snapshot-b")
    assert baseline == manifest(start_label="Renamed result")
    assert baseline == manifest(code_label="Renamed code step")
    assert baseline == manifest(input_label="Renamed input")


def test_workspace_manifest_ignores_file_loader_metadata() -> None:
    def manifest_for(serialized_state: dict):
        return workspace_code_trust_manifest(
            _workspace_manifest_from_attrs(
                {"itool_state": json.dumps(serialized_state)}
            )
        )

    first = manifest_for(_serialized_imagetool_state())
    changed = manifest_for(_serialized_imagetool_state(target="pathlib:Path.unlink"))

    assert not first.has_executable_code
    assert first.canonical_bytes() == changed.canonical_bytes()


def test_workspace_code_trust_manifest_does_not_decode_unrelated_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _workspace_manifest("ax.plot([1])")
    attrs = manifest["nodes"][0]["payload_attrs"]
    attrs.extend(workspace_format._workspace_manifest_attrs({"preview": np.arange(4)}))

    def fail_if_decoded(*_args, **_kwargs):
        raise AssertionError("unrelated array metadata was decoded")

    monkeypatch.setattr(workspace_format, "_workspace_decode_array", fail_if_decoded)

    assert workspace_code_trust_manifest(manifest).has_executable_code


def test_workspace_code_trust_manifest_rejects_array_in_code_attribute_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _workspace_manifest("ax.plot([1])")
    manifest["nodes"][0]["payload_attrs"] = workspace_format._workspace_manifest_attrs(
        {
            "tool_cls_qualname": (
                "erlab.interactive._figurecomposer._tool:FigureComposerTool"
            ),
            "tool_state": np.arange(4),
        }
    )

    def fail_if_decoded(*_args, **_kwargs):
        raise AssertionError("array metadata was decoded")

    monkeypatch.setattr(workspace_format, "_workspace_decode_array", fail_if_decoded)

    with pytest.raises(TypeError, match="must be a string"):
        workspace_code_trust_manifest(manifest)


def test_workspace_code_trust_manifest_fails_closed_without_payload_attrs() -> None:
    with pytest.raises(TypeError, match="payload attributes"):
        workspace_code_trust_manifest(
            {"nodes": [{"kind": "tool", "path": "figures/0"}]}
        )


def test_trusted_workspace_location_is_checked_before_manifest_building(
    monkeypatch, manager_context
) -> None:
    monkeypatch.setattr(
        workspace_trust, "workspace_path_is_trusted", lambda _path: True
    )

    def fail_if_built(*_args, **_kwargs):
        raise AssertionError("trusted-location load built the code manifest")

    monkeypatch.setattr(workspace_trust, "workspace_code_trust_manifest", fail_if_built)
    with manager_context() as manager:
        trust = manager._workspace_controller._loaded_workspace_code_trust(
            "trusted.itws", {"nodes": []}, selected_paths=None
        )

    assert document_trust_is_trusted(trust)


def test_workspace_host_routes_materialized_tool_trust_to_one_state(
    qtbot, manager_context
) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    with manager_context() as manager:
        manager.add_imagetool(ImageTool(data), show=False)
        manager._workspace_state.code_trust = untrusted_document_trust()

        tool = _TrustProbeTool(data)
        qtbot.addWidget(tool)
        add_source_childtool(manager, tool, 0, show=False)

        assert not document_trust_has_trusted_lineage(tool._current_document_trust())

        manager._workspace_state.code_trust = new_document_trust()
        manager._workspace_controller._refresh_code_trust_ui()

        assert document_trust_has_trusted_lineage(tool._current_document_trust())


def test_workspace_host_preserves_identical_saved_signature(
    qtbot, manager_context
) -> None:
    manifest = create_manifest(
        workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN,
        workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION,
        (create_entry("test.code", "tools/0/code", "run_code()"),),
    )
    reset_saved_code_trust(domain=workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN)
    try:
        save_document_trust(new_document_trust(), manifest)
        signed = load_document_trust(manifest)
        assert _trust_uses_saved_signature(signed)

        data = xr.DataArray(np.arange(3.0), dims="x", name="data")
        with manager_context() as manager:
            manager._workspace_state.code_trust = signed
            tool = _TrustProbeTool(data)
            qtbot.addWidget(tool)
            tool.set_document_trust(signed, notify=False)

            manager._workspace_controller._configure_tool_code_trust(
                tool, location_getter=lambda: "tools/0"
            )

            assert _trust_uses_saved_signature(manager._workspace_state.code_trust)
    finally:
        reset_saved_code_trust(domain=workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN)


def test_workspace_local_code_edit_commits_only_after_success(
    manager_context, monkeypatch
) -> None:
    saved_entry = create_entry("test.code", "tools/0/code", "saved()")
    edited_entry = create_entry("test.code", "tools/0/code", "edited()")
    signed = _document_trust_after_save(
        new_document_trust(),
        create_manifest(
            workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN,
            workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION,
            (saved_entry,),
        ),
        saved_trusted_lineage=True,
        signature_stored=True,
    )

    with manager_context() as manager:
        manager._workspace_state.code_trust = signed
        controller = manager._workspace_controller
        scan_calls = 0
        original_manifest_builder = (
            workspace_trust.current_workspace_code_trust_manifest
        )

        def tracked_manifest_builder(manager):
            nonlocal scan_calls
            scan_calls += 1
            return original_manifest_builder(manager)

        monkeypatch.setattr(
            workspace_trust,
            "current_workspace_code_trust_manifest",
            tracked_manifest_builder,
        )
        controller._mark_workspace_dirty(structure="test presentation edit")
        assert manager._workspace_state.code_trust == signed

        def fail_edit() -> None:
            with controller.local_code_edit(
                (edited_entry,),
                edited_entries=(edited_entry,),
            ) as capability:
                assert manager._workspace_state.code_trust == signed
                assert (
                    controller.issue_code_execution_capability((edited_entry,))
                    is capability
                )
                raise RuntimeError("validation failed")

        with pytest.raises(RuntimeError, match="validation failed"):
            fail_edit()

        assert manager._workspace_state.code_trust == signed
        assert scan_calls == 0

        with controller.local_code_edit(
            (edited_entry,),
            edited_entries=(edited_entry,),
        ):
            assert manager._workspace_state.code_trust == signed

        assert manager._workspace_state.code_trust != signed
        assert scan_calls == 1
        assert document_trust_has_trusted_lineage(
            manager._workspace_state.code_trust
        ), (
            manager._workspace_state.code_trust.local_document_identities,
            tuple(
                entry.document_identity()
                for entry in current_workspace_code_trust_manifest(manager).entries
            ),
        )


@pytest.mark.parametrize(
    "trust",
    [
        pytest.param(new_document_trust(), id="local-lineage"),
        pytest.param(trusted_location_document_trust(), id="trusted-location"),
    ],
)
def test_trusted_workspace_local_code_edit_does_not_scan_manifest(
    manager_context,
    monkeypatch,
    trust,
) -> None:
    edited_entry = create_entry("test.code", "tools/0/code", "edited()")

    with manager_context() as manager:
        manager._workspace_state.code_trust = trust
        controller = manager._workspace_controller

        def unexpected_manifest_scan(_manager):
            raise AssertionError("trusted local edit scanned the workspace manifest")

        monkeypatch.setattr(
            workspace_trust,
            "current_workspace_code_trust_manifest",
            unexpected_manifest_scan,
        )

        with controller.local_code_edit(
            (edited_entry,),
            edited_entries=(edited_entry,),
        ) as capability:
            assert execution_capability_allows(capability, (edited_entry,))

        assert manager._workspace_state.code_trust == trust


def test_workspace_partial_capability_requires_explicit_graph_request(
    manager_context,
) -> None:
    local_entry = create_entry("test.code", "tools/local/code", "local()")
    external_entry = create_entry(
        "test.code",
        "tools/external/code",
        "external()",
    )

    with manager_context() as manager:
        controller = manager._workspace_controller
        controller._set_workspace_code_trust(
            untrusted_document_trust(
                create_manifest(
                    workspace_trust.WORKSPACE_CODE_TRUST_DOMAIN,
                    workspace_trust.WORKSPACE_CODE_TRUST_POLICY_VERSION,
                    (external_entry,),
                )
            )
        )

        def check_open_transaction() -> None:
            with controller.local_code_edit(
                (local_entry, external_entry),
                edited_entries=(local_entry,),
            ) as partial_capability:
                assert partial_capability is not None
                assert (
                    controller.issue_code_execution_capability(
                        (local_entry, external_entry)
                    )
                    is None
                )
                assert (
                    controller.issue_code_execution_capability(
                        (local_entry, external_entry),
                        allow_partial=True,
                    )
                    is partial_capability
                )
                assert execution_capability_allows(
                    partial_capability,
                    (local_entry,),
                )
                assert not execution_capability_allows(
                    partial_capability,
                    (external_entry,),
                )
                raise RuntimeError("stop test transaction")

        with pytest.raises(RuntimeError, match="stop test transaction"):
            check_open_transaction()


def test_signed_workspace_runs_locally_edited_provenance_without_review(
    qtbot,
    manager_context,
) -> None:
    source = xr.DataArray(np.arange(3.0), dims="x", name="data")
    original = full_data(
        ScriptCodeOperation(label="Offset", code="derived = derived + 1")
    )
    candidate = full_data(
        ScriptCodeOperation(label="Offset", code="derived = derived + 2")
    )
    tool = ImageTool(source + 1)
    qtbot.addWidget(tool)

    with manager_context() as manager:
        manager.add_imagetool(
            tool,
            show=False,
            provenance_spec=original,
            replay_source_data=source,
        )
        manifest = current_workspace_code_trust_manifest(manager)
        signed = _document_trust_after_save(
            new_document_trust(),
            manifest,
            saved_trusted_lineage=True,
            signature_stored=True,
        )
        manager._workspace_controller._set_workspace_code_trust(signed)

        node = manager._node_for_target(0)
        manager._provenance_edit_controller._validate_and_replace(
            node,
            "display",
            candidate,
        )

        xr.testing.assert_identical(node.current_public_data(), source + 2)
        assert document_trust_has_trusted_lineage(manager._workspace_state.code_trust)
        assert not _trust_uses_saved_signature(manager._workspace_state.code_trust)


def test_local_provenance_edit_does_not_authorize_equal_code_at_another_node(
    qtbot, manager_context
) -> None:
    source = xr.DataArray(np.arange(3.0), dims="x", name="data")
    shared = full_data(
        ScriptCodeOperation(label="Shared", code="derived = derived + 1")
    )
    different = full_data(
        ScriptCodeOperation(label="Different", code="derived = derived + 2")
    )
    first = ImageTool(source + 1)
    second = ImageTool(source + 2)
    qtbot.addWidget(first)
    qtbot.addWidget(second)

    with manager_context() as manager:
        manager.add_imagetool(
            first,
            show=False,
            provenance_spec=shared,
            replay_source_data=source,
        )
        manager.add_imagetool(
            second,
            show=False,
            provenance_spec=different,
            replay_source_data=source,
        )
        manifest = current_workspace_code_trust_manifest(manager)
        manager._workspace_controller._set_workspace_code_trust(
            untrusted_document_trust(manifest)
        )

        second_node = manager._node_for_target(1)
        manager._provenance_edit_controller._validate_and_replace(
            second_node,
            "display",
            shared,
        )

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert manager.code_trust_banner.isVisible()
        xr.testing.assert_identical(second_node.current_public_data(), source + 1)
        runtime_entries = provenance_code_trust_entries(
            shared,
            location_prefix="runtime",
        )
        assert second.slicer_area._stored_code_authorizer is not None
        assert second.slicer_area._stored_code_authorizer(runtime_entries) is not None
        assert first.slicer_area._stored_code_authorizer is not None
        assert first.slicer_area._stored_code_authorizer(runtime_entries) is None

        first_local = full_data(
            ScriptCodeOperation(label="First local", code="derived = derived + 3")
        )
        first_node = manager._node_for_target(0)
        manager._provenance_edit_controller._validate_and_replace(
            first_node,
            "display",
            first_local,
        )

        assert document_trust_has_trusted_lineage(manager._workspace_state.code_trust)
        xr.testing.assert_identical(first_node.current_public_data(), source + 3)


def test_local_source_provenance_edit_owns_source_and_display_locations(
    qtbot, manager_context
) -> None:
    source = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("y", "x"),
        name="data",
    )
    shared = _model_fit_source_spec()
    different_operation = _model_fit_operation().model_copy(
        update={
            "parameters": {
                "c0": _ModelFitParameterSpec(value=0.0),
                "c1": _ModelFitParameterSpec(expr="3 * c0"),
            }
        }
    )
    different = full_data(different_operation)
    external = ImageTool(source)
    parent = ImageTool(source)
    child = ImageTool(source)
    for tool in (external, parent, child):
        qtbot.addWidget(tool)

    with manager_context() as manager:
        manager.add_imagetool(
            external,
            show=False,
            provenance_spec=shared,
            replay_source_data=source,
        )
        manager.add_imagetool(parent, show=False)
        child_uid = manager.add_imagetool_child(
            child,
            1,
            show=False,
            source_spec=different,
        )
        manager._workspace_controller._set_workspace_code_trust(
            untrusted_document_trust(current_workspace_code_trust_manifest(manager))
        )

        child_node = manager._child_node(child_uid)
        manager._provenance_edit_controller._validate_and_replace(
            child_node,
            "source",
            shared,
        )

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        runtime_entries = provenance_code_trust_entries(
            shared,
            location_prefix="runtime",
        )
        assert child.slicer_area._stored_code_authorizer is not None
        assert child.slicer_area._stored_code_authorizer(runtime_entries) is not None
        assert external.slicer_area._stored_code_authorizer is not None
        assert external.slicer_area._stored_code_authorizer(runtime_entries) is None


def test_workspace_host_merges_added_tool_trust(qtbot, manager_context) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    recipe = FigureRecipeState(
        operations=(
            FigureOperationState.custom(label="custom", code="ax.set_title('code')"),
        )
    )
    tool = FigureComposerTool(data, recipe=recipe)
    qtbot.addWidget(tool)
    tool.set_document_trust(untrusted_document_trust(), notify=False)

    with manager_context() as manager:
        manager.add_figuretool(tool, show=False)

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert not document_trust_has_trusted_lineage(tool._document_trust)
        assert current_workspace_code_trust_manifest(manager).has_executable_code


def test_local_figure_added_to_mixed_workspace_keeps_exact_local_ownership(
    qtbot, manager_context
) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    recipe = FigureRecipeState(
        operations=(
            FigureOperationState.custom(label="custom", code="ax.set_title('same')"),
        )
    )
    external = FigureComposerTool(data, recipe=recipe)
    local = FigureComposerTool(data, recipe=recipe)
    qtbot.addWidget(external)
    qtbot.addWidget(local)
    external_manifest = external._current_code_trust_manifest()
    assert external_manifest is not None
    external.set_document_trust(
        untrusted_document_trust(external_manifest), notify=False
    )

    with manager_context() as manager:
        manager.add_figuretool(external, show=False)
        manager.add_figuretool(local, show=False)

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert manager.code_trust_banner.isVisible()
        figure_rendering._render_into_figure(
            external, external.figure, sync_visible=False
        )
        figure_rendering._render_into_figure(local, local.figure, sync_visible=False)
        assert external.figure.axes[0].get_title() == ""
        assert local.figure.axes[0].get_title() == "same"


def test_local_edit_does_not_authorize_equal_code_at_another_node_location(
    qtbot, manager_context
) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    first = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            operations=(
                FigureOperationState.custom(
                    label="first", code="ax.set_title('shared')"
                ),
            )
        ),
    )
    second = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            operations=(
                FigureOperationState.custom(
                    label="second", code="ax.set_title('external second')"
                ),
            )
        ),
    )
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    for tool in (first, second):
        manifest = tool._current_code_trust_manifest()
        assert manifest is not None
        tool.set_document_trust(untrusted_document_trust(manifest), notify=False)

    with manager_context() as manager:
        manager.add_figuretool(first, show=False)
        manager.add_figuretool(second, show=False)
        assert not document_trust_is_trusted(manager._workspace_state.code_trust)

        second.tool_status = second.tool_status.model_copy(
            update={
                "operations": (
                    FigureOperationState.custom(
                        label="second", code="ax.set_title('shared')"
                    ),
                )
            }
        )

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert manager.code_trust_banner.isVisible()
        qtbot.waitUntil(
            lambda: second.figure.axes[0].get_title() == "shared", timeout=1000
        )
        figure_rendering._render_into_figure(first, first.figure, sync_visible=False)
        assert first.figure.axes[0].get_title() == ""
        first_manifest = first._current_code_trust_manifest()
        assert first_manifest is not None
        assert (
            manager._workspace_controller.issue_code_execution_capability(
                first_manifest.entries
            )
            is None
        )

        first.tool_status = first.tool_status.model_copy(
            update={
                "operations": (
                    FigureOperationState.custom(
                        label="first", code="ax.set_title('local first')"
                    ),
                )
            }
        )

        assert document_trust_has_trusted_lineage(manager._workspace_state.code_trust)
        assert not manager.code_trust_banner.isVisible()
        qtbot.waitUntil(
            lambda: first.figure.axes[0].get_title() == "local first", timeout=1000
        )
        qtbot.waitUntil(
            lambda: second.figure.axes[0].get_title() == "shared", timeout=1000
        )


def test_workspace_host_merges_added_imagetool_trust(qtbot, manager_context) -> None:
    restored = _restore_imagetool(qtbot, state=_serialized_imagetool_state())

    assert restored.slicer_area._load_func is None
    with manager_context() as manager:
        manager.add_imagetool(restored, show=False)

        assert document_trust_is_trusted(manager._workspace_state.code_trust)
        assert restored.slicer_area._load_func is None
        manifest = current_workspace_code_trust_manifest(manager)
        assert not manifest.has_executable_code


@pytest.mark.parametrize("attachment", ["root", "child"])
def test_workspace_host_imports_standalone_executable_provenance_as_external(
    qtbot, manager_context, attachment: str
) -> None:
    provenance = _script_source(
        "import os\nderived = data + int(os.path.exists(os.devnull))"
    )
    restored = _restore_imagetool(
        qtbot, attrs={"itool_provenance_spec": provenance.model_dump_json()}
    )

    with manager_context() as manager:
        if attachment == "root":
            target = manager.add_imagetool(restored, show=False)
        else:
            parent = ImageTool(xr.DataArray(np.arange(3.0), dims="x"))
            qtbot.addWidget(parent)
            manager.add_imagetool(parent, show=False)
            target = manager.add_imagetool_child(restored, 0, show=False)

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert restored.provenance_spec == provenance
        assert manager._node_for_target(target).displayed_provenance_spec == provenance
        manifest = current_workspace_code_trust_manifest(manager)
    assert [entry.feature for entry in manifest.entries] == [
        "erlab.provenance.script-code"
    ]


def test_current_workspace_manifest_includes_imagetool_live_source_code(
    qtbot,
    manager_context,
) -> None:
    data = xr.DataArray(np.arange(4.0), dims="x", coords={"x": np.arange(4.0)})
    root = ImageTool(data)
    child = ImageTool(data)
    qtbot.addWidget(root)
    qtbot.addWidget(child)

    with manager_context() as manager:
        manager.add_imagetool(root, show=False)
        manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=_model_fit_source_spec(),
            source_state="stale",
        )

        assert document_trust_is_trusted(manager._workspace_state.code_trust)
        manifest = current_workspace_code_trust_manifest(manager)

    source_entries = [
        entry for entry in manifest.entries if "/source/provenance/" in entry.location
    ]
    assert [entry.code for entry in source_entries] == ["2 * c0"]


def test_workspace_host_retains_tool_payload_verification_failure(
    qtbot, manager_context
) -> None:
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    with manager_context() as manager:
        manager.add_imagetool(ImageTool(data), show=False)
        tool = _TrustProbeTool(data)
        qtbot.addWidget(tool)
        tool.set_document_trust(untrusted_document_trust(), notify=False)

        manager._workspace_state.code_trust = trusted_location_document_trust()
        add_source_childtool(manager, tool, 0, show=False)

        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert not document_trust_is_trusted(tool._current_document_trust())
        assert not document_trust_has_trusted_lineage(tool._document_trust)


def test_workspace_approval_persists_and_restores_figure_execution(
    qtbot, tmp_path, manager_context, monkeypatch
) -> None:
    reset_saved_code_trust(domain="erlab.workspace")
    workspace_path = tmp_path / "code-trust.itws"
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(
            FigureOperationState.custom(label="custom", code="ax.set_title('trusted')"),
        ),
        primary_source="data",
    )

    with manager_context() as manager:
        figure_uid = manager.add_figuretool(
            FigureComposerTool(data, recipe=recipe), show=False
        )
        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        assert _trust_uses_saved_signature(manager._workspace_state.code_trust)
        assert _trust_uses_saved_signature(
            load_document_trust(current_workspace_code_trust_manifest(manager))
        )

    reset_saved_code_trust(domain="erlab.workspace")
    with manager_context() as manager:
        manager.add_figuretool(FigureComposerTool(data), show=False)
        assert _load_workspace(manager, workspace_path, replace=False)
        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        untrusted_copy = tmp_path / "untrusted-copy.itws"
        manager._workspace_controller.saving._save_workspace_document(untrusted_copy)
        assert not _trust_uses_saved_signature(
            load_document_trust(current_workspace_code_trust_manifest(manager))
        )

    with manager_context() as manager:
        assert _load_workspace(manager, workspace_path)
        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
        assert not manager.code_trust_banner.isHidden()
        node = manager._child_node(figure_uid)
        assert node.materialize_pending_workspace_payload()
        tool = node.tool_window
        assert isinstance(tool, FigureComposerTool)
        figure_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
        assert tool.figure.axes[0].get_title() == ""

        monkeypatch.setattr(
            "erlab.interactive.imagetool.manager._workspace._controller."
            "confirm_code_trust",
            lambda *_args, **_kwargs: True,
        )
        manager._workspace_controller.review_and_approve_workspace_code_trust()
        qtbot.waitUntil(
            lambda: tool.figure.axes[0].get_title() == "trusted", timeout=1000
        )
        manager._workspace_controller.saving._save_workspace_document(workspace_path)

    with manager_context() as manager:
        assert _load_workspace(manager, workspace_path)
        assert _trust_uses_saved_signature(manager._workspace_state.code_trust)
        signed = manager._workspace_state.code_trust
        node = manager._child_node(figure_uid)
        assert node.materialize_pending_workspace_payload()
        assert manager._workspace_state.code_trust == signed


def test_workspace_save_uses_snapshot_trust_without_overwriting_current_state(
    tmp_path, manager_context
) -> None:
    reset_saved_code_trust(domain="erlab.workspace")
    untrusted_snapshot = _workspace_manifest(
        "ax.plot([1])", workspace_id="untrusted-snapshot"
    )
    trusted_snapshot = _workspace_manifest(
        "ax.plot([2])", workspace_id="trusted-snapshot"
    )

    with manager_context() as manager:
        controller = manager._workspace_controller
        manager._workspace_state.code_trust = new_document_trust()

        controller._record_saved_workspace_code_trust(
            untrusted_snapshot,
            trusted_lineage=False,
            current_document=True,
        )

        untrusted_manifest = workspace_code_trust_manifest(untrusted_snapshot)
        assert not _trust_uses_saved_signature(load_document_trust(untrusted_manifest))
        assert document_trust_is_trusted(manager._workspace_state.code_trust)

        manager._workspace_state.code_trust = untrusted_document_trust()
        snapshot = controller.saving._workspace_generation_save_snapshot(
            manager._workspace_state.dirty_generation,
            fname=tmp_path / "snapshot.itws",
        )
        try:
            assert not snapshot.trusted_lineage
        finally:
            snapshot.close()
        controller._record_saved_workspace_code_trust(
            trusted_snapshot,
            trusted_lineage=True,
            current_document=False,
        )

        assert _trust_uses_saved_signature(
            load_document_trust(workspace_code_trust_manifest(trusted_snapshot))
        )
        assert not document_trust_is_trusted(manager._workspace_state.code_trust)
