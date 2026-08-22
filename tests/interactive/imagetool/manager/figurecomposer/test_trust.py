from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from qtpy import QtWidgets

import erlab.interactive._figurecomposer._rendering as figurecomposer_rendering
import erlab.interactive.utils
from erlab.interactive._code_trust import (
    create_manifest,
    document_trust_has_trusted_lineage,
    document_trust_is_trusted,
    external_document_trust,
    new_document_trust,
    untrusted_document_trust,
)
from erlab.interactive._code_trust._api import _document_trust_after_save
from erlab.interactive._code_trust._application import (
    load_document_trust,
    reset_saved_code_trust,
)
from erlab.interactive._figurecomposer import FigureComposerTool, FigureSourceState
from erlab.interactive._figurecomposer._model._state import (
    FigureMethodFamily,
    FigureOperationState,
    FigureRecipeState,
)
from erlab.interactive._figurecomposer._trust import (
    figure_code_trust_entries,
    figure_operation_execution_entries,
)


def _custom_recipe(code: str = "pass") -> FigureRecipeState:
    return FigureRecipeState(
        operations=(FigureOperationState.custom(label="custom", code=code),)
    )


def _custom_tool(qtbot, code: str = "pass") -> FigureComposerTool:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(2.0), dims="x"), recipe=_custom_recipe(code)
    )
    qtbot.addWidget(tool)
    return tool


def _manifest_bytes(recipe: FigureRecipeState) -> bytes:
    return create_manifest(
        "erlab.workspace",
        1,
        figure_code_trust_entries(recipe, location_prefix="figures/0"),
    ).canonical_bytes()


def test_figure_code_trust_entries_capture_code_and_execution_context() -> None:
    custom = FigureOperationState.custom(label="custom", code="ax.plot([1, 2])")
    method = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="text",
    ).model_copy(
        update={
            "enabled": False,
            "method_transform": "data",
            "method_transform_expression": "ax.transAxes",
        }
    )

    entries = figure_code_trust_entries(
        FigureRecipeState(operations=(custom, method)), location_prefix="figures/0"
    )

    assert [entry.feature for entry in entries] == [
        "erlab.figure-composer.custom-code",
        "erlab.figure-composer.custom-transform",
    ]
    assert entries[0].code == "ax.plot([1, 2])"
    assert entries[0].location == "figures/0/operations/0"
    assert entries[1].context["enabled"] is False
    assert entries[1].context["transform"] == "data"


def test_figure_code_trust_entries_ignore_empty_code() -> None:
    assert not figure_code_trust_entries(
        _custom_recipe(""), location_prefix="figures/0"
    )


def test_figure_code_trust_entries_ignore_source_provenance_containers() -> None:
    recipe = FigureRecipeState(
        sources=(
            FigureSourceState(
                name="data",
                label="data",
                provenance_spec={"kind": "script"},
            ),
        )
    )

    entries = figure_code_trust_entries(recipe, location_prefix="figure")

    assert entries == ()


def test_figure_operation_execution_entries_only_include_executed_code() -> None:
    custom = FigureOperationState.custom(label="custom", code="pass")
    preset_transform = FigureOperationState.method(
        family=FigureMethodFamily.AXES, name="plot"
    ).model_copy(update={"method_transform_expression": "ax.transAxes"})
    custom_transform = preset_transform.model_copy(
        update={"method_transform": "custom"}
    )

    recipe = FigureRecipeState(operations=(custom, preset_transform, custom_transform))

    assert figure_operation_execution_entries(recipe, 0, location_prefix="figure")
    assert not figure_operation_execution_entries(recipe, 1, location_prefix="figure")
    assert figure_operation_execution_entries(recipe, 2, location_prefix="figure")


@pytest.mark.parametrize(
    "operation",
    [
        FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="set_title",
        ).model_copy(
            update={
                "method_transform": "custom",
                "method_transform_expression": "ax.transAxes",
            }
        ),
        FigureOperationState.custom(label="custom", code="").model_copy(
            update={
                "method_transform": "custom",
                "method_transform_expression": "ax.transAxes",
            }
        ),
    ],
)
def test_figure_code_trust_ignores_unreachable_transform_expression(
    operation: FigureOperationState,
) -> None:
    recipe = FigureRecipeState(operations=(operation,))

    assert not figure_code_trust_entries(recipe, location_prefix="figure")
    assert not figure_operation_execution_entries(recipe, 0, location_prefix="figure")


def test_invalid_operation_does_not_request_code_authorization(
    qtbot, monkeypatch
) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(3.0), dims="x", name="data"),
        recipe=FigureRecipeState(
            operations=(
                FigureOperationState.custom(label="custom", code="raise RuntimeError"),
            )
        ),
    )
    qtbot.addWidget(tool)
    monkeypatch.setattr(
        tool.operation_editor,
        "has_input_error",
        lambda _operation: True,
    )
    monkeypatch.setattr(
        tool,
        "_authorize_code_execution",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid operation requested code authorization"
        ),
    )
    monkeypatch.setattr(
        tool,
        "_issue_code_execution_capability",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid operation requested an execution capability"
        ),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    figurecomposer_rendering._render_into_figure(
        tool,
        tool.figure,
        sync_visible=False,
    )
    tool.export_figure()


def test_figure_render_keeps_capability_at_execution_boundary(
    qtbot, monkeypatch
) -> None:
    tool = _custom_tool(qtbot, "ax.set_title('guarded')")
    original_execute = figurecomposer_rendering.execute_with_capability
    guarded_calls: list[object] = []

    def execute_with_capability(capability, entries, execute):
        assert capability is not None
        guarded_calls.append(capability)
        return original_execute(capability, entries, execute)

    monkeypatch.setattr(
        figurecomposer_rendering,
        "execute_with_capability",
        execute_with_capability,
    )

    figurecomposer_rendering._render_into_figure(
        tool,
        tool.figure,
        sync_visible=False,
    )

    assert guarded_calls
    assert tool.figure.axes[0].get_title() == "guarded"


def test_figure_operation_discards_legacy_trusted_field() -> None:
    operation = FigureOperationState.model_validate(
        {
            "kind": "custom",
            "label": "custom",
            "code": "pass",
            "trusted": True,
        }
    )

    assert "trusted" not in operation.model_dump()


def test_figure_manifest_tracks_execution_controls_but_not_review_labels() -> None:
    first = FigureOperationState.custom(label="First label", code="ax.plot([1])")
    second = FigureOperationState.custom(label="Second", code="ax.plot([2])")

    def canonical(*operations: FigureOperationState) -> bytes:
        return _manifest_bytes(FigureRecipeState(operations=operations))

    baseline = canonical(first, second)
    assert baseline == canonical(first.model_copy(update={"label": "Renamed"}), second)
    assert baseline != canonical(
        first.model_copy(update={"code": "ax.plot([3])"}), second
    )
    assert baseline != canonical(second, first)
    assert baseline != canonical(first.model_copy(update={"enabled": False}), second)
    assert baseline != canonical(
        first.model_copy(update={"sources": ("alternate",)}), second
    )


def test_figure_manifest_ignores_safe_rendering_pipeline_state() -> None:
    setup_title = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="set_title",
        args=("safe title",),
    )
    custom = FigureOperationState.custom(
        label="custom",
        code="result = ax.get_title()",
    )

    recipe = FigureRecipeState(operations=(setup_title, custom))
    baseline = _manifest_bytes(recipe)

    assert baseline == _manifest_bytes(
        recipe.model_copy(
            update={
                "operations": (
                    setup_title.model_copy(update={"method_args": ("changed",)}),
                    custom,
                )
            }
        )
    )
    assert baseline == _manifest_bytes(
        recipe.model_copy(
            update={"setup": recipe.setup.model_copy(update={"figsize": (8.0, 4.0)})}
        )
    )
    assert baseline == _manifest_bytes(
        recipe.model_copy(
            update={
                "operations": (
                    setup_title.model_copy(update={"label": "Renamed"}),
                    custom,
                )
            }
        )
    )
    assert baseline == _manifest_bytes(
        recipe.model_copy(
            update={"export": recipe.export.model_copy(update={"dpi": 300.0})}
        )
    )


def test_figure_manifest_tracks_custom_code_namespace() -> None:
    operation = FigureOperationState.custom(label="custom", code="data.plot(ax=ax)")
    source = FigureSourceState(name="data", label="First label")

    recipe = FigureRecipeState(sources=(source,), operations=(operation,))
    baseline = _manifest_bytes(recipe)

    assert baseline == _manifest_bytes(
        recipe.model_copy(
            update={"sources": (source.model_copy(update={"label": "Renamed"}),)}
        )
    )
    assert baseline != _manifest_bytes(
        recipe.model_copy(
            update={"sources": (source.model_copy(update={"name": "other"}),)}
        )
    )
    assert baseline != _manifest_bytes(
        recipe.model_copy(
            update={"setup": recipe.setup.model_copy(update={"ncols": 2})}
        )
    )


def test_figure_manifest_tracks_custom_transform_mode() -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES, name="text"
    ).model_copy(
        update={
            "method_transform": "custom",
            "method_transform_expression": "ax.transAxes",
        }
    )
    changed = operation.model_copy(update={"method_transform": "data"})

    assert figure_code_trust_entries(
        FigureRecipeState(operations=(operation,)), location_prefix="figures/0"
    ) != figure_code_trust_entries(
        FigureRecipeState(operations=(changed,)), location_prefix="figures/0"
    )


@pytest.mark.parametrize("initial_reason", ["signature", "no_code", "untrusted"])
def test_local_figure_code_edit_never_requires_review(
    qtbot,
    initial_reason: str,
) -> None:
    tool = FigureComposerTool(
        xr.DataArray(np.arange(3.0), dims="x", name="data"),
        recipe=(
            FigureRecipeState()
            if initial_reason == "no_code"
            else _custom_recipe("ax.set_title('before')")
        ),
    )
    qtbot.addWidget(tool)
    manifest = tool._current_code_trust_manifest()
    assert manifest is not None
    if initial_reason == "signature":
        trust = _document_trust_after_save(
            new_document_trust(),
            manifest,
            saved_trusted_lineage=True,
            signature_stored=True,
        )
    elif initial_reason == "no_code":
        trust = external_document_trust(manifest)
    else:
        trust = untrusted_document_trust(manifest)
    tool.set_document_trust(trust)

    tool.tool_status = tool.tool_status.model_copy(
        update={
            "operations": (
                FigureOperationState.custom(
                    label="custom",
                    code="ax.set_title('locally edited')",
                ),
            )
        }
    )

    assert document_trust_has_trusted_lineage(tool._document_trust)
    assert tool.figure.axes[0].get_title() == "locally edited"


def test_local_figure_edit_authorizes_only_edited_identity(qtbot) -> None:
    external_first = FigureOperationState.custom(
        label="first", code="ax.set_title('external first')"
    )
    external_second = FigureOperationState.custom(
        label="second", code="ax.set_title('external second')"
    )
    tool = FigureComposerTool(
        xr.DataArray(np.arange(3.0), dims="x", name="data"),
        recipe=FigureRecipeState(operations=(external_first, external_second)),
    )
    qtbot.addWidget(tool)
    manifest = tool._current_code_trust_manifest()
    assert manifest is not None
    tool.set_document_trust(untrusted_document_trust(manifest), notify=False)

    local_first = FigureOperationState.custom(
        label="first", code="ax.set_title('local first')"
    )
    tool.tool_status = tool.tool_status.model_copy(
        update={"operations": (local_first, external_second)}
    )

    assert not document_trust_is_trusted(tool._document_trust)
    qtbot.waitUntil(
        lambda: (
            bool(tool.figure.axes) and tool.figure.axes[0].get_title() == "local first"
        ),
        timeout=1000,
    )
    assert not tool._authorize_code_execution(
        figure_operation_execution_entries(
            tool.tool_status, 1, location_prefix="figure"
        )
    )

    local_second = FigureOperationState.custom(
        label="second", code="ax.set_title('local second')"
    )
    tool.tool_status = tool.tool_status.model_copy(
        update={"operations": (local_first, local_second)}
    )

    assert document_trust_has_trusted_lineage(tool._document_trust)
    qtbot.waitUntil(
        lambda: (
            bool(tool.figure.axes) and tool.figure.axes[0].get_title() == "local second"
        ),
        timeout=1000,
    )


def test_local_figure_reorder_changes_saved_signature_to_local_lineage(qtbot) -> None:
    recipe = FigureRecipeState(
        operations=(
            FigureOperationState.custom(label="first", code="ax.set_title('first')"),
            FigureOperationState.custom(label="second", code="ax.set_title('second')"),
        )
    )
    tool = FigureComposerTool(xr.DataArray(np.arange(3.0), dims="x"), recipe=recipe)
    qtbot.addWidget(tool)
    manifest = tool._current_code_trust_manifest()
    assert manifest is not None
    signed = _document_trust_after_save(
        new_document_trust(),
        manifest,
        saved_trusted_lineage=True,
        signature_stored=True,
    )
    tool.set_document_trust(signed)

    tool.tool_status = tool.tool_status.model_copy(
        update={"operations": tuple(reversed(tool.tool_status.operations))}
    )

    assert tool._document_trust != signed
    assert document_trust_has_trusted_lineage(tool._document_trust)


def test_figure_file_trust_is_durable_and_untrusted_saves_do_not_sign(
    qtbot, tmp_path
) -> None:
    reset_saved_code_trust(domain="erlab.figure-composer-file")
    data = xr.DataArray(np.arange(3.0), dims="x", name="data")
    recipe = FigureRecipeState(
        sources=(FigureSourceState(name="data", label="data"),),
        operations=(
            FigureOperationState.custom(label="custom", code="ax.set_title('ok')"),
        ),
        primary_source="data",
    )
    tool = FigureComposerTool(data, recipe=recipe)
    qtbot.addWidget(tool)
    trusted_file = tmp_path / "trusted-figure.h5"
    tool.to_file(trusted_file)

    restored = erlab.interactive.utils.ToolWindow.from_file(trusted_file)
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    assert document_trust_has_trusted_lineage(restored._document_trust)

    reset_saved_code_trust(domain="erlab.figure-composer-file")
    untrusted = erlab.interactive.utils.ToolWindow.from_file(trusted_file)
    qtbot.addWidget(untrusted)
    assert isinstance(untrusted, FigureComposerTool)
    assert not document_trust_has_trusted_lineage(untrusted._document_trust)
    duplicate = untrusted.duplicate()
    qtbot.addWidget(duplicate)
    assert not document_trust_has_trusted_lineage(duplicate._document_trust)

    untrusted_file = tmp_path / "untrusted-figure.h5"
    untrusted.to_file(untrusted_file)
    manifest = untrusted._current_code_trust_manifest()
    assert manifest is not None
    assert not document_trust_is_trusted(load_document_trust(manifest))


def test_failed_figure_file_save_does_not_sign(qtbot, tmp_path) -> None:
    reset_saved_code_trust(domain="erlab.figure-composer-file")
    tool = _custom_tool(qtbot)
    manifest = tool._current_code_trust_manifest()
    assert manifest is not None

    with pytest.raises(FileNotFoundError):
        tool.to_file(tmp_path / "missing" / "figure.h5")

    assert not document_trust_is_trusted(load_document_trust(manifest))


def test_tool_file_is_not_written_when_trust_manifest_build_fails(
    qtbot, tmp_path, monkeypatch
) -> None:
    tool = _custom_tool(qtbot)
    target = tmp_path / "figure.h5"

    def fail_manifest_build(_cls, _status, _attrs):
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(
        FigureComposerTool,
        "_code_trust_manifest_from_saved_metadata",
        classmethod(fail_manifest_build),
    )

    with pytest.raises(RuntimeError, match="manifest failed"):
        tool.to_file(target)

    assert not target.exists()


def test_save_refreshes_banner_after_removing_all_executable_code(
    qtbot, tmp_path
) -> None:
    reset_saved_code_trust(domain="erlab.figure-composer-file")
    source_path = tmp_path / "source.h5"
    tool = _custom_tool(qtbot)
    tool.to_file(source_path)

    reset_saved_code_trust(domain="erlab.figure-composer-file")
    restored = erlab.interactive.utils.ToolWindow.from_file(source_path)
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    assert not document_trust_has_trusted_lineage(restored._document_trust)
    assert not restored._code_trust_banner.isHidden()

    restored.tool_status = restored.tool_status.model_copy(update={"operations": ()})
    restored.to_file(tmp_path / "without-code.h5")

    assert document_trust_has_trusted_lineage(restored._document_trust)
    assert restored._code_trust_banner.isHidden()


def test_review_clears_trust_warning_after_all_code_is_removed(
    qtbot, monkeypatch
) -> None:
    tool = _custom_tool(qtbot)
    tool.set_document_trust(untrusted_document_trust())
    tool.tool_status = tool.tool_status.model_copy(update={"operations": ()})
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected dialog")),
    )

    tool._review_code_trust()

    assert document_trust_has_trusted_lineage(tool._document_trust)
    assert tool._code_trust_banner.isHidden()
