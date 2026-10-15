import dataclasses

import matplotlib.colors as mcolors
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._figurecomposer._rendering as rendering
from erlab.interactive._code_trust import approve_document_trust
from erlab.interactive._figurecomposer import (
    FigureComposerTool,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
)
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._fit_sources import FigureFitSourceSnapshot
from erlab.interactive.utils import ToolWindow, qt_is_valid
from tests.interactive.imagetool.manager.helpers import _exec_generated_code

from ._common import _select_operation_rows


@pytest.fixture
def fit_snapshot():
    import lmfit.models

    x = np.linspace(-2.0, 2.0, 19)
    y = 2.4 * np.exp(-(((x - 0.2) / 0.6) ** 2))
    y += 0.04 * np.sin(np.arange(x.size))
    y[7] = np.nan
    data = xr.DataArray(y, dims="energy", coords={"energy": x})
    result = data.xlm.modelfit(
        "energy",
        model=lmfit.models.GaussianModel(),
        nan_policy="omit",
        weights=xr.full_like(data, 1.0 / 0.06),
    )
    return FigureFitSourceSnapshot.from_dataset(result)


@pytest.fixture
def component_snapshot():
    from lmfit.models import GaussianModel, LinearModel

    x = np.linspace(-2.0, 2.0, 31)
    model = GaussianModel(prefix="peak_") + LinearModel(prefix="bg_")
    params = model.make_params(
        peak_amplitude=2.5,
        peak_center=0.2,
        peak_sigma=0.4,
        bg_intercept=0.3,
        bg_slope=0.1,
    )
    y = model.eval(params, x=x) + 0.01 * np.sin(np.arange(x.size))
    y[9] = np.nan
    data = xr.DataArray(y, dims="energy", coords={"energy": x})
    result = data.xlm.modelfit(
        "energy", model=model, params=params, guess=False, nan_policy="omit"
    )
    return FigureFitSourceSnapshot.from_dataset(result)


@pytest.fixture
def replacement_component_snapshot(component_snapshot):
    from lmfit.models import GaussianModel, LinearModel

    model = GaussianModel(prefix="other_") + LinearModel(prefix="bg_")
    params = model.make_params(
        other_amplitude=2.5,
        other_center=0.2,
        other_sigma=0.4,
        bg_intercept=0.3,
        bg_slope=0.1,
    )
    result = component_snapshot.native_dataset.modelfit_data.xlm.modelfit(
        "energy", model=model, params=params, guess=False, nan_policy="omit"
    )
    return FigureFitSourceSnapshot.from_dataset(result)


def _tool(qtbot, snapshot, operation):
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": snapshot},
        sources=(FigureSourceState(name="edc_fit", kind="fit"),),
        operations=(operation,),
    )
    qtbot.addWidget(tool)
    return tool


def _render(tool):
    rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert not tool._operation_render_errors
    return tool.figure.axes[0]


def _style_control(parent, widget_type, name):
    component = parent.property("componentName")
    if component is not None:
        for prefix in (
            "figureComposerFitPlotComponentLine",
            "figureComposerFitPlotComponentBand",
        ):
            if name.startswith(prefix):
                name = f"{prefix}:{component}:" + name.removeprefix(prefix)
                break
    return parent.findChild(widget_type, name)


def _set_text(parent, name, value):
    control = _style_control(parent, QtWidgets.QLineEdit, name)
    assert control is not None
    control.setText(value)
    control.setModified(True)
    control.editingFinished.emit()


def _activate_combo(parent, name, value):
    combo = next(
        combo
        for combo in parent.findChildren(QtWidgets.QComboBox, name)
        if not combo.signalsBlocked()
    )
    index = combo.findData(value)
    if index < 0:
        index = combo.findText(value)
    assert index >= 0
    combo.setCurrentIndex(index)
    combo.activated.emit(index)
    return combo


def _component_page(editor, name):
    _activate_combo(editor.current_page, "figureComposerFitPlotComponent", name)
    stack = editor.current_page.findChild(
        QtWidgets.QStackedWidget, "figureComposerFitPlotComponentStyles"
    )
    assert stack is not None
    page = stack.currentWidget()
    assert page.property("componentName") == name
    return page


def _set_kwargs(page, prefix, text):
    _set_text(page, prefix + "KwargsEdit", text)


def _assert_equal_artists(actual, expected):
    assert len(actual.lines) == len(expected.lines)
    for left, right in zip(actual.lines, expected.lines, strict=True):
        np.testing.assert_allclose(left.get_xdata(), right.get_xdata(), equal_nan=True)
        np.testing.assert_allclose(left.get_ydata(), right.get_ydata(), equal_nan=True)
        assert mcolors.to_rgba(left.get_color()) == mcolors.to_rgba(right.get_color())
        assert left.get_marker() == right.get_marker()
        assert left.get_linestyle() == right.get_linestyle()
        assert left.get_linewidth() == right.get_linewidth()
        assert left.get_markersize() == right.get_markersize()
        assert left.get_markeredgewidth() == right.get_markeredgewidth()
        assert left.get_label() == right.get_label()
    assert len(actual.collections) == len(expected.collections)
    for left, right in zip(actual.collections, expected.collections, strict=True):
        np.testing.assert_allclose(left.get_facecolor(), right.get_facecolor())
        np.testing.assert_allclose(left.get_edgecolor(), right.get_edgecolor())
        np.testing.assert_allclose(left.get_linewidths(), right.get_linewidths())
        assert left.get_alpha() == right.get_alpha()
        assert left.get_label() == right.get_label()
        assert len(left.get_paths()) == len(right.get_paths())
        for left_path, right_path in zip(
            left.get_paths(), right.get_paths(), strict=True
        ):
            np.testing.assert_allclose(left_path.vertices, right_path.vertices)


@pytest.mark.parametrize("numpoints", [None, 67])
def test_fit_components_render_codegen_and_restore(
    qtbot, component_snapshot, numpoints
):
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "show_components": True,
            "show_data": False,
            "show_fit": False,
            "fit_numpoints": numpoints,
            "fit_sigma": 0.95,
            "fit_band_kw": {"alpha": 0.3, "edgecolor": (0.0, 0.0, 0.5, 1.0)},
            "fit_component_kw": {
                "peak_": {"color": "magenta", "ls": ":", "label": "Peak"},
                "bg_": {"color": "black", "linewidth": 2.5, "dashes": (2, 1)},
            },
            "fit_component_band_kw": {
                "peak_": {"label": "Peak uncertainty", "alpha": 0.45},
                "bg_": {
                    "label": "Background uncertainty",
                    "hatch": "/",
                    "edgecolor": (0.0, 0.0, 0.0, 1.0),
                },
            },
        }
    )
    tool = _tool(qtbot, component_snapshot, operation)
    original = _render(tool)
    assert len(original.lines) == 2
    assert original.lines[0].get_label() == "Peak"
    assert original.lines[1].get_label() == "bg_"
    assert original.lines[0].get_xdata().size == (numpoints or 31)
    assert np.isnan(original.lines[0].get_ydata()[9]) == (numpoints is None)
    assert len(original.collections) == 3
    for band, line, alpha, label in zip(
        original.collections[1:],
        original.lines,
        (0.45, 0.3),
        ("Peak uncertainty", "Background uncertainty"),
        strict=True,
    ):
        np.testing.assert_allclose(
            band.get_facecolor(), [mcolors.to_rgba(line.get_color(), alpha=alpha)]
        )
        assert band.get_label() == label
        assert len(band.get_paths()) == (2 if numpoints is None else 1)
    assert component_snapshot.component_names == ("peak_", "bg_")
    namespace = _exec_generated_code(
        tool.generated_code(), {"edc_fit": component_snapshot.result_dataset()}
    )
    _assert_equal_artists(original, namespace["fig"].axes[0])
    saved = tool.to_dataset()
    restored = ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)
    restored.set_document_trust(
        approve_document_trust(
            restored._document_trust, restored._current_code_trust_manifest()
        ),
        notify=False,
    )
    assert restored.tool_status.operations == tool.tool_status.operations
    assert (
        restored.source_payload("edc_fit", FigureFitSourceSnapshot).component_names
        == component_snapshot.component_names
    )
    _assert_equal_artists(original, _render(restored))


def test_fit_components_editor_updates_visibility_and_styles(qtbot, component_snapshot):
    tool = _tool(
        qtbot,
        component_snapshot,
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={"fit_sigma": 0.95}
        ),
    )
    editor = tool.operation_editor
    editor.select_section("fit_data")
    errors = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowYerr"
    )
    assert errors is None
    assert (
        editor.current_page.findChild(
            QtWidgets.QLineEdit, "figureComposerFitPlotErrorCapsize"
        )
        is None
    )
    editor.select_section("fit_components")
    show = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowComponents"
    )
    assert show is not None
    assert show.isEnabled()
    assert not show.isChecked()
    settings = editor.current_page.findChild(
        QtWidgets.QWidget, "figureComposerFitPlotComponentSettings"
    )
    assert settings is not None
    assert settings.isHidden()
    show.setChecked(True)
    assert tool.tool_status.operations[0].show_components
    assert not settings.isHidden()
    page = _component_page(editor, "peak_")
    _set_text(page, "figureComposerFitPlotComponentLineColor", "red")
    _set_text(page, "figureComposerFitPlotComponentLineLinewidth", "2.5")
    _set_text(page, "figureComposerFitPlotComponentLineLabel", "Peak")
    _set_text(page, "figureComposerFitPlotComponentBandLabel", "Peak uncertainty")
    _set_text(page, "figureComposerFitPlotComponentBandColor", "cyan")
    _set_text(page, "figureComposerFitPlotComponentBandEdgecolor", "navy")
    _set_text(page, "figureComposerFitPlotComponentBandAlpha", "0.4")
    operation = tool.tool_status.operations[0]
    assert operation.fit_component_kw == {
        "peak_": {"color": "red", "linewidth": 2.5, "label": "Peak"}
    }
    assert operation.fit_component_band_kw == {
        "peak_": {
            "label": "Peak uncertainty",
            "facecolor": "cyan",
            "edgecolor": "navy",
            "alpha": 0.4,
        }
    }
    axis = _render(tool)
    lines = axis.lines
    assert lines[-2].get_color() == "red"
    assert lines[-2].get_linewidth() == 2.5
    assert lines[-2].get_label() == "Peak"
    assert axis.collections[-2].get_label() == "Peak uncertainty"
    assert axis.collections[-2].get_alpha() == 0.4
    np.testing.assert_allclose(
        axis.collections[-2].get_facecolor(), [mcolors.to_rgba("cyan", alpha=0.4)]
    )
    np.testing.assert_allclose(
        axis.collections[-2].get_edgecolor(), [mcolors.to_rgba("navy", alpha=0.4)]
    )
    _set_text(page, "figureComposerFitPlotComponentLineLinewidth", "invalid")
    assert tool.tool_status.operations[0].fit_component_kw == operation.fit_component_kw
    assert editor.has_input_error(tool.tool_status.operations[0])
    _set_text(page, "figureComposerFitPlotComponentLineLinewidth", "")
    assert tool.tool_status.operations[0].fit_component_kw == {
        "peak_": {"color": "red", "label": "Peak"}
    }
    assert not editor.has_input_error(tool.tool_status.operations[0])
    other = _component_page(editor, "bg_")
    assert (
        _style_control(
            other, QtWidgets.QLineEdit, "figureComposerFitPlotComponentLineLabel"
        ).text()
        == ""
    )
    assert (
        _style_control(
            other, QtWidgets.QLineEdit, "figureComposerFitPlotComponentBandLabel"
        ).text()
        == ""
    )
    styled = tool.tool_status.operations[0]
    show.setChecked(False)
    assert settings.isHidden()
    assert not tool.tool_status.operations[0].show_components
    show.setChecked(True)
    assert not settings.isHidden()
    assert tool.tool_status.operations[0].fit_component_kw == styled.fit_component_kw
    assert (
        tool.tool_status.operations[0].fit_component_band_kw
        == styled.fit_component_band_kw
    )


def test_fit_components_editor_hiding_discards_invalid_input(qtbot, component_snapshot):
    tool = _tool(
        qtbot,
        component_snapshot,
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={
                "show_components": True,
                "fit_component_kw": {"peak_": {"linewidth": 2.5}},
            }
        ),
    )
    editor = tool.operation_editor
    editor.select_section("fit_components")
    page = _component_page(editor, "peak_")
    _set_text(page, "figureComposerFitPlotComponentLineLinewidth", "invalid")
    assert editor.has_input_error(tool.tool_status.operations[0])
    show = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowComponents"
    )
    show.setChecked(False)
    assert not editor.has_input_error(tool.tool_status.operations[0])
    assert tool.tool_status.operations[0].fit_component_kw == {
        "peak_": {"linewidth": 2.5}
    }
    assert len(_render(tool).lines) == 2
    qtbot.wait_until(lambda: not qt_is_valid(show))
    show = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowComponents"
    )
    show.setChecked(True)
    page = _component_page(editor, "peak_")
    width = _style_control(
        page, QtWidgets.QLineEdit, "figureComposerFitPlotComponentLineLinewidth"
    )
    assert width.text() == "2.5"
    assert not editor.has_input_error(tool.tool_status.operations[0])


@pytest.mark.parametrize(
    ("prefix", "suffix", "invalid", "valid"),
    [
        ("figureComposerFitPlotComponentLine", "Linewidth", "invalid", "2.5"),
        ("figureComposerFitPlotComponentBand", "Linewidth", "invalid", "2.5"),
        ("figureComposerFitPlotComponentLine", "KwargsEdit", "[", "rasterized=True"),
        ("figureComposerFitPlotComponentBand", "KwargsEdit", "[", "hatch='//'"),
    ],
)
def test_fit_component_input_errors_are_independent(
    qtbot, component_snapshot, prefix, suffix, invalid, valid
):
    tool = _tool(
        qtbot,
        component_snapshot,
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={"show_components": True}
        ),
    )
    editor = tool.operation_editor
    editor.select_section("fit_components")
    first = _component_page(editor, "peak_")
    first_control = _style_control(first, QtWidgets.QLineEdit, prefix + suffix)
    _set_text(first, prefix + suffix, invalid)
    assert editor.has_input_error(tool.tool_status.operations[0])
    second = _component_page(editor, "bg_")
    second_control = _style_control(second, QtWidgets.QLineEdit, prefix + suffix)
    assert first_control.objectName() != second_control.objectName()
    _set_text(second, prefix + suffix, valid)
    assert editor.has_input_error(tool.tool_status.operations[0])
    assert first_control.text() == invalid
    _set_text(first, prefix + suffix, valid)
    assert not editor.has_input_error(tool.tool_status.operations[0])


def test_fit_source_switch_reconciles_each_selected_operation(
    qtbot, component_snapshot, replacement_component_snapshot
):
    operations = tuple(
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={
                "show_components": True,
                "fit_component_kw": {
                    "peak_": {"color": "red"},
                    "bg_": {"label": label},
                },
                "fit_component_band_kw": {
                    "peak_": {"label": "Peak uncertainty"},
                    "bg_": {"alpha": 0.4},
                },
            }
        )
        for label in ("Background 1", "Background 2")
    )
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": component_snapshot, "other_fit": replacement_component_snapshot},
        sources=tuple(
            FigureSourceState(name=name, kind="fit")
            for name in ("edc_fit", "other_fit")
        ),
        operations=operations,
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0, 1))
    editor = tool.operation_editor
    editor.select_section("sources")
    assert len(editor.binding.editable_operation_ids()) == 2
    _activate_combo(
        editor.source_controls, "figureComposerFitPlotSourceCombo", "other_fit"
    )
    for original, updated in zip(operations, tool.tool_status.operations, strict=True):
        assert updated.sources == ("other_fit",)
        assert updated.fit_component_kw == {"bg_": original.fit_component_kw["bg_"]}
        assert updated.fit_component_band_kw == {"bg_": {"alpha": 0.4}}
    _render(tool)
    tool.undo()
    assert tool.tool_status.operations == operations
    tool.redo()
    assert all(
        operation.sources == ("other_fit",) for operation in tool.tool_status.operations
    )
    _render(tool)


@pytest.mark.parametrize("refresh", ["switch", "replace", "linked_add"])
def test_fit_source_changes_discard_only_removed_component_input_errors(
    qtbot, component_snapshot, replacement_component_snapshot, refresh
):
    source = FigureSourceState(
        name="edc_fit", kind="fit", node_uid="fit-node", provider_id="fit"
    )
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": component_snapshot, "other_fit": replacement_component_snapshot},
        sources=(
            source,
            FigureSourceState(name="other_fit", kind="fit"),
        ),
        operations=(
            FigureOperationState.fit_plot(source="edc_fit").model_copy(
                update={"show_components": True}
            ),
            FigureOperationState.fit_plot(source="other_fit"),
        ),
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0,))
    editor = tool.operation_editor
    editor.select_section("fit_components")
    page = _component_page(editor, "peak_")
    _set_text(page, "figureComposerFitPlotComponentLineLinewidth", "invalid")
    _set_kwargs(page, "figureComposerFitPlotComponentBand", "[")
    page = _component_page(editor, "bg_")
    _set_text(page, "figureComposerFitPlotComponentLineLinewidth", "invalid")
    editor.select_section("fit_data")
    _set_text(editor.current_page, "figureComposerFitPlotDataLinewidth", "invalid")
    color = editor.current_page.findChild(
        QtWidgets.QLineEdit, "figureComposerFitPlotDataColor"
    )
    editor.record_input_error(color.parentWidget(), FigureComposerInputError("color"))
    other_operation = tool.tool_status.operations[1]
    editor.set_input_errors(
        editor._input_errors | {other_operation.operation_id: {"unrelated": "error"}}
    )
    editor.select_section("sources")
    _activate_combo(
        editor.source_controls, "figureComposerFitPlotSourceCombo", "edc_fit"
    )
    assert editor.has_input_error(tool.tool_status.operations[0])
    generation = editor._generation
    if refresh == "switch":
        _activate_combo(
            editor.source_controls, "figureComposerFitPlotSourceCombo", "other_fit"
        )
    elif refresh == "replace":
        assert tool.replace_source_payload(
            "edc_fit", source, replacement_component_snapshot
        )
    else:
        incoming = source.model_copy(update={"name": "incoming_fit"})
        result = tool.add_source_payloads(
            (incoming,), {"incoming_fit": replacement_component_snapshot}
        )
        assert result.updated == (("incoming_fit", "edc_fit"),)
    qtbot.wait_until(lambda: editor._generation > generation)
    operation = tool.tool_status.operations[0]
    assert set(editor._input_errors[operation.operation_id]) == {
        "figureComposerFitPlotComponentLine:bg_:Linewidth",
        "figureComposerFitPlotDataLinewidth",
        "figureComposerFitPlotDataColor",
    }
    assert editor._input_errors[other_operation.operation_id] == {"unrelated": "error"}
    assert editor.has_input_error(tool.tool_status.operations[0])
    editor.select_section("fit_components")
    page = _component_page(editor, "bg_")
    _set_text(page, "figureComposerFitPlotComponentLineLinewidth", "2")
    editor.select_section("fit_data")
    _set_text(editor.current_page, "figureComposerFitPlotDataLinewidth", "2")
    _set_text(editor.current_page, "figureComposerFitPlotDataColor", "red")
    assert not editor.has_input_error(tool.tool_status.operations[0])
    editor.clear_input_errors((other_operation.operation_id,))
    _render(tool)


@pytest.mark.parametrize("refresh", ["switch", "replace", "linked_add"])
def test_fit_source_changes_discard_unavailable_error_bar_input_errors(
    qtbot, fit_snapshot, component_snapshot, refresh
):
    source = FigureSourceState(
        name="edc_fit", kind="fit", node_uid="fit-node", provider_id="fit"
    )
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": fit_snapshot, "other_fit": component_snapshot},
        sources=(source, FigureSourceState(name="other_fit", kind="fit")),
        operations=(FigureOperationState.fit_plot(source="edc_fit"),),
    )
    qtbot.addWidget(tool)
    editor = tool.operation_editor
    tool.show_figure_window(activate=False)
    editor.select_section("fit_data")
    _set_text(editor.current_page, "figureComposerFitPlotErrorCapsize", "invalid")
    assert editor.has_input_error(tool.tool_status.operations[0])
    generation = editor._generation
    editor._begin_transient_combo_popup_guard()
    if refresh == "switch":
        editor.select_section("sources")
        _activate_combo(
            editor.source_controls, "figureComposerFitPlotSourceCombo", "other_fit"
        )
    elif refresh == "replace":
        assert tool.replace_source_payload("edc_fit", source, component_snapshot)
    else:
        incoming = source.model_copy(update={"name": "incoming_fit"})
        result = tool.add_source_payloads(
            (incoming,), {"incoming_fit": component_snapshot}
        )
        assert result.updated == (("incoming_fit", "edc_fit"),)
    qtbot.wait_until(lambda: editor._generation > generation)
    assert not editor.has_input_error(tool.tool_status.operations[0])
    editor.select_section("fit_data")
    assert (
        editor.current_page.findChild(
            QtWidgets.QLineEdit, "figureComposerFitPlotErrorCapsize"
        )
        is None
    )
    qtbot.wait_until(
        lambda: (
            len(tool.figure.axes[0].lines) == 2
            and len(tool.figure.axes[0].lines[0].get_ydata())
            == component_snapshot.native_dataset.modelfit_data.size
        )
    )
    _render(tool)


def test_fit_input_errors_survive_incompatible_batch_selection(qtbot, fit_snapshot):
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": fit_snapshot},
        sources=(FigureSourceState(name="edc_fit", kind="fit"),),
        operations=(
            FigureOperationState.fit_plot(source="edc_fit"),
            FigureOperationState.custom(label="Code", code="ax.set_title('fit')"),
        ),
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0,))
    editor = tool.operation_editor
    editor.select_section("fit_data")
    _set_text(editor.current_page, "figureComposerFitPlotErrorCapsize", "invalid")
    operation = tool.tool_status.operations[0]
    errors = editor._input_errors[operation.operation_id].copy()
    _select_operation_rows(tool, (0, 1))
    assert editor._input_errors[operation.operation_id] == errors
    _select_operation_rows(tool, (0,))
    assert editor._input_errors[operation.operation_id] == errors


def test_fit_batch_source_switch_reconciles_each_operations_input_errors(
    qtbot, fit_snapshot, component_snapshot, replacement_component_snapshot
):
    snapshots = {
        "weighted_fit": fit_snapshot,
        "component_fit": component_snapshot,
        "replacement_fit": replacement_component_snapshot,
    }
    tool = FigureComposerTool.from_fit_sources(
        snapshots,
        sources=tuple(FigureSourceState(name=name, kind="fit") for name in snapshots),
        operations=tuple(
            FigureOperationState.fit_plot(source=name).model_copy(
                update={"show_components": True}
            )
            for name in ("weighted_fit", "component_fit")
        ),
    )
    qtbot.addWidget(tool)
    editor = tool.operation_editor
    _select_operation_rows(tool, (0,))
    editor.select_section("fit_data")
    _set_text(editor.current_page, "figureComposerFitPlotErrorCapsize", "invalid")
    _select_operation_rows(tool, (1,))
    editor.select_section("fit_components")
    _set_text(
        _component_page(editor, "peak_"),
        "figureComposerFitPlotComponentLineLinewidth",
        "invalid",
    )
    original_errors = {
        operation_id: errors.copy()
        for operation_id, errors in editor._input_errors.items()
    }
    _select_operation_rows(tool, (0, 1))
    assert editor._input_errors == original_errors
    editor.select_section("sources")
    generation = editor._generation
    _activate_combo(
        editor.source_controls,
        "figureComposerFitPlotSourceCombo",
        "replacement_fit",
    )
    qtbot.wait_until(lambda: editor._generation > generation)
    assert all(
        not editor.has_input_error(operation)
        for operation in tool.tool_status.operations
    )
    _render(tool)


def test_fit_batch_hiding_components_discards_each_components_input_errors(
    qtbot, fit_snapshot, component_snapshot
):
    snapshots = {"first_fit": fit_snapshot, "second_fit": component_snapshot}
    tool = FigureComposerTool.from_fit_sources(
        snapshots,
        sources=tuple(FigureSourceState(name=name, kind="fit") for name in snapshots),
        operations=tuple(
            FigureOperationState.fit_plot(source=name).model_copy(
                update={"show_components": True}
            )
            for name in snapshots
        ),
    )
    qtbot.addWidget(tool)
    editor = tool.operation_editor
    for index, component in enumerate(("gaussian", "peak_")):
        _select_operation_rows(tool, (index,))
        editor.select_section("fit_components")
        _set_text(
            _component_page(editor, component),
            "figureComposerFitPlotComponentLineLinewidth",
            "invalid",
        )
    _select_operation_rows(tool, (0, 1))
    assert all(
        editor.has_input_error(operation) for operation in tool.tool_status.operations
    )
    editor.select_section("fit_components")
    show = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowComponents"
    )
    generation = editor._generation
    show.setChecked(False)
    qtbot.wait_until(lambda: editor._generation > generation)
    assert all(
        not editor.has_input_error(operation)
        for operation in tool.tool_status.operations
    )
    assert len(_render(tool).lines) == 4


@pytest.mark.parametrize("refresh", ["replace", "linked_add"])
def test_fit_source_refresh_discards_unselected_operations_unavailable_inputs(
    qtbot, fit_snapshot, component_snapshot, refresh
):
    source = FigureSourceState(
        name="edc_fit", kind="fit", node_uid="fit-node", provider_id="fit"
    )
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": fit_snapshot},
        sources=(source,),
        operations=tuple(
            FigureOperationState.fit_plot(source="edc_fit") for _index in range(2)
        ),
    )
    qtbot.addWidget(tool)
    tool.show_figure_window(activate=False)
    editor = tool.operation_editor
    _select_operation_rows(tool, (1,))
    editor.select_section("fit_data")
    _set_text(editor.current_page, "figureComposerFitPlotErrorCapsize", "invalid")
    _select_operation_rows(tool, (0,))
    unselected = tool.tool_status.operations[1]
    assert editor.has_input_error(unselected)
    editor._begin_transient_combo_popup_guard()
    generation = editor._generation
    if refresh == "replace":
        assert tool.replace_source_payload("edc_fit", source, component_snapshot)
    else:
        incoming = source.model_copy(update={"name": "incoming_fit"})
        result = tool.add_source_payloads(
            (incoming,), {"incoming_fit": component_snapshot}
        )
        assert result.updated == (("incoming_fit", "edc_fit"),)
    qtbot.wait_until(lambda: editor._generation > generation)
    assert not editor.has_input_error(unselected)
    qtbot.wait_until(lambda: len(tool.figure.axes[0].lines) == 4)
    assert all(
        len(line.get_ydata()) == component_snapshot.native_dataset.modelfit_data.size
        for line in tool.figure.axes[0].lines
    )
    _render(tool)


def test_fit_source_reconciliation_preserves_styles_without_component_metadata(
    qtbot, component_snapshot
):
    snapshot = dataclasses.replace(component_snapshot, component_names=())
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "fit_component_kw": {"legacy": {"color": "red"}},
            "fit_component_band_kw": {"legacy": {"label": "Legacy uncertainty"}},
        }
    )
    tool = _tool(qtbot, snapshot, operation)
    assert tool.tool_status.operations[0].fit_component_kw == operation.fit_component_kw
    assert (
        tool.tool_status.operations[0].fit_component_band_kw
        == operation.fit_component_band_kw
    )


@pytest.mark.parametrize("refresh", ["replace", "linked_add"])
def test_fit_source_refresh_reconciles_styles_and_history(
    qtbot, component_snapshot, replacement_component_snapshot, refresh
):
    source = FigureSourceState(
        name="edc_fit", kind="fit", node_uid="fit-node", provider_id="fit"
    )
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "show_components": True,
            "fit_sigma": 1.0,
            "fit_component_kw": {
                "peak_": {"color": "red"},
                "bg_": {"label": "Background"},
            },
            "fit_component_band_kw": {
                "peak_": {"alpha": 0.3},
                "bg_": {"label": "Background uncertainty"},
            },
        }
    )
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": component_snapshot}, sources=(source,), operations=(operation,)
    )
    qtbot.addWidget(tool)
    if refresh == "replace":
        assert tool.replace_source_payload(
            "edc_fit", source, replacement_component_snapshot
        )
    else:
        incoming = source.model_copy(update={"name": "incoming_fit"})
        result = tool.add_source_payloads(
            (incoming,), {"incoming_fit": replacement_component_snapshot}
        )
        assert result.updated == (("incoming_fit", "edc_fit"),)
    updated = tool.tool_status.operations[0]
    assert updated.fit_component_kw == {"bg_": {"label": "Background"}}
    assert updated.fit_component_band_kw == {"bg_": {"label": "Background uncertainty"}}
    assert (
        tool.source_payload("edc_fit", FigureFitSourceSnapshot)
        is replacement_component_snapshot
    )
    axis = _render(tool)
    namespace = _exec_generated_code(
        tool.generated_code(),
        {"edc_fit": replacement_component_snapshot.result_dataset()},
    )
    _assert_equal_artists(axis, namespace["fig"].axes[0])
    tool.undo()
    assert tool.tool_status.operations[0] == operation
    assert tool.source_payload("edc_fit", FigureFitSourceSnapshot) is component_snapshot
    _render(tool)
    tool.redo()
    assert tool.tool_status.operations[0] == updated
    assert (
        tool.source_payload("edc_fit", FigureFitSourceSnapshot)
        is replacement_component_snapshot
    )
    restored = ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert restored.tool_status.operations[0] == updated
    restored.set_document_trust(
        approve_document_trust(
            restored._document_trust, restored._current_code_trust_manifest()
        ),
        notify=False,
    )
    _assert_equal_artists(_render(tool), _render(restored))


def test_fit_components_editor_mixed_visibility(qtbot, component_snapshot):
    operations = tuple(
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={"show_components": visible}
        )
        for visible in (True, False)
    )
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": component_snapshot},
        sources=(FigureSourceState(name="edc_fit", kind="fit"),),
        operations=operations,
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0, 1))
    editor = tool.operation_editor
    editor.select_section("fit_components")
    settings = editor.current_page.findChild(
        QtWidgets.QWidget, "figureComposerFitPlotComponentSettings"
    )
    show = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowComponents"
    )
    assert show.checkState() == QtCore.Qt.CheckState.PartiallyChecked
    assert settings.isHidden()
    show.setCheckState(QtCore.Qt.CheckState.Checked)
    assert not settings.isHidden()
    assert all(operation.show_components for operation in tool.tool_status.operations)
    show.setChecked(False)
    assert settings.isHidden()
    assert not any(
        operation.show_components for operation in tool.tool_status.operations
    )


@pytest.mark.parametrize("numpoints", [None, 67])
def test_fit_components_preserve_multipeak_names_in_editor_and_code(qtbot, numpoints):
    from erlab.analysis.fit.models import MultiPeakModel

    x = np.linspace(-1.0, 1.0, 41)
    model = MultiPeakModel(
        npeaks=2,
        peak_shapes=["gaussian"],
        fd=False,
        convolve=False,
        background="constant",
    )
    params = model.make_params(
        p0_height=2.0,
        p0_center=-0.3,
        p0_width=0.2,
        p1_height=1.0,
        p1_center=0.3,
        p1_width=0.2,
        const_bkg=0.2,
    )
    y = model.eval(params, x=x) + 0.01 * np.sin(np.arange(x.size))
    result = xr.DataArray(y, dims="x", coords={"x": x}).xlm.modelfit(
        "x", model=model, params=params, guess=False
    )
    styles = {"2Peak_p0": {"color": "red", "label": "Peak 1"}}
    snapshot = FigureFitSourceSnapshot.from_dataset(result)
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "show_components": True,
            "fit_component_kw": styles,
            "fit_numpoints": numpoints,
            "fit_sigma": 2.0,
        }
    )
    tool = _tool(qtbot, snapshot, operation)
    editor = tool.operation_editor
    editor.select_section("fit_components")
    page = _component_page(editor, "2Peak_p0")
    color = _style_control(
        page, QtWidgets.QLineEdit, "figureComposerFitPlotComponentLineColor"
    )
    assert color is not None
    assert color.text() == "red"
    assert (
        _style_control(
            page, QtWidgets.QLineEdit, "figureComposerFitPlotComponentLineLabel"
        ).text()
        == "Peak 1"
    )
    color.editingFinished.emit()
    assert not editor.has_input_error(tool.tool_status.operations[0])
    original = _render(tool)
    namespace = _exec_generated_code(
        tool.generated_code(), {"edc_fit": snapshot.result_dataset()}
    )
    _assert_equal_artists(original, namespace["fig"].axes[0])
    assert original.lines[-3].get_label() == "Peak 1"
    assert len(original.collections) == 4
    fitted = result.modelfit_results.item()
    component_x = original.lines[-3].get_xdata()
    expected = fitted.eval_components(x=component_x)
    for line, values in zip(original.lines[-3:], expected.values(), strict=True):
        np.testing.assert_allclose(line.get_ydata(), values)
    restored = ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    restored.set_document_trust(
        approve_document_trust(
            restored._document_trust, restored._current_code_trust_manifest()
        ),
        notify=False,
    )
    _assert_equal_artists(original, _render(restored))


def test_fit_component_editor_uses_safe_metadata_and_preserves_styles(
    qtbot, component_snapshot, monkeypatch
):
    def blocked_decode(_self):
        raise AssertionError("Component controls must use stored metadata")

    monkeypatch.setattr(FigureFitSourceSnapshot, "result_dataset", blocked_decode)
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "fit_component_kw": {
                "peak_": {"rasterized": True},
                "old_component": {"color": "green"},
            },
            "fit_component_band_kw": {"old_component": {"hatch": "/"}},
        }
    )
    tool = _tool(qtbot, component_snapshot, operation)
    editor = tool.operation_editor
    editor.select_section("fit_components")
    selector = editor.current_page.findChild(
        QtWidgets.QComboBox, "figureComposerFitPlotComponent"
    )
    assert selector is not None
    assert [selector.itemText(index) for index in range(selector.count())] == [
        "peak_",
        "bg_",
    ]
    page = _component_page(editor, "peak_")
    _set_text(page, "figureComposerFitPlotComponentLineLabel", "Peak")
    updated = tool.tool_status.operations[0]
    assert updated.fit_component_kw == {
        "peak_": {"rasterized": True, "label": "Peak"},
    }
    assert updated.fit_component_band_kw == {}


def test_fit_components_require_model_execution_trust(
    qtbot, component_snapshot, monkeypatch
):
    from erlab.interactive._code_trust import untrusted_document_trust
    from erlab.interactive._figurecomposer._trust import (
        figure_operation_execution_entries,
        figure_source_entries,
    )

    tool = _tool(
        qtbot,
        component_snapshot,
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={"show_components": True}
        ),
    )
    expected = figure_source_entries(
        {"edc_fit": component_snapshot}, location_prefix="figure"
    )
    assert expected
    assert (
        figure_operation_execution_entries(
            tool.tool_status,
            0,
            location_prefix="figure",
            source_payloads=tool.source_payloads(),
        )
        == expected
    )
    tool.set_document_trust(
        untrusted_document_trust(tool._current_code_trust_manifest()), notify=False
    )

    def blocked_decode(_self):
        raise AssertionError("An untrusted fit must not decode its model")

    monkeypatch.setattr(FigureFitSourceSnapshot, "result_dataset", blocked_decode)
    assert not _render(tool).lines


@pytest.mark.parametrize("sigma", [None, 0.95, 1.0, 2.0, 3.0, 1e-9, 1000.0])
def test_fit_plot_sigma_state_validation(sigma):
    state = FigureOperationState.fit_plot(source="edc_fit").model_dump()
    state["fit_sigma"] = sigma
    assert FigureOperationState.model_validate(state).fit_sigma == sigma


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_fit_plot_sigma_state_rejects_invalid(sigma):
    import pydantic

    state = FigureOperationState.fit_plot(source="edc_fit").model_dump()
    state["fit_sigma"] = sigma
    with pytest.raises(pydantic.ValidationError):
        FigureOperationState.model_validate(state)


@pytest.mark.parametrize("numpoints", [None, 83])
@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0, 0.95])
def test_fit_plot_generated_code_matches_render(qtbot, fit_snapshot, numpoints, sigma):
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "fit_numpoints": numpoints,
            "fit_sigma": sigma,
            "fit_show_yerr": True,
            "fit_data_kw": {"marker": "s", "color": "black", "markersize": 4},
            "fit_line_kw": {"color": "magenta", "linestyle": "--", "linewidth": 2.3},
            "fit_band_kw": {"color": "cyan", "alpha": 0.35, "hatch": "/"},
        }
    )
    tool = _tool(qtbot, fit_snapshot, operation)
    rendered = _render(tool)
    assert not tool._operation_issues(tool.tool_status.operations[0])
    namespace = _exec_generated_code(
        tool.generated_code(), {"edc_fit": fit_snapshot.result_dataset()}
    )
    _assert_equal_artists(rendered, namespace["fig"].axes[0])
    assert rendered.lines[0].get_xdata().size == (numpoints or 19)
    assert rendered.lines[1].get_xdata().size == 19
    assert len(rendered.collections) == 2
    if numpoints is None:
        assert np.isnan(rendered.lines[0].get_ydata()[7])
    else:
        assert np.isfinite(rendered.lines[0].get_ydata()).all()


@pytest.mark.parametrize("numpoints", [None, 67])
@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0, 0.95])
def test_fit_plot_roundtrip_retains_model_and_options(
    qtbot, fit_snapshot, numpoints, sigma
):
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "fit_numpoints": numpoints,
            "fit_sigma": sigma,
            "fit_show_yerr": True,
            "fit_data_kw": {"color": "red", "marker": "+"},
            "fit_line_kw": {"color": "navy", "drawstyle": "steps-mid"},
            "fit_band_kw": {"alpha": 0.42, "edgecolor": "purple"},
        }
    )
    tool = _tool(qtbot, fit_snapshot, operation)
    original = _render(tool)
    saved = tool.to_dataset()
    tool.close()
    restored = ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    # GaussianModel includes saved expressions for its derived parameters.
    restored.set_document_trust(
        approve_document_trust(
            restored._document_trust, restored._current_code_trust_manifest()
        ),
        notify=False,
    )
    assert restored.tool_status.operations == tool.tool_status.operations
    validated = FigureRecipeState.model_validate_json(
        tool.tool_status.model_dump_json()
    )
    assert validated.operations == tool.tool_status.operations
    _assert_equal_artists(_render(restored), original)
    assert not restored.source_data()
    xr.testing.assert_identical(
        restored.source_payload("edc_fit", FigureFitSourceSnapshot).native_dataset,
        fit_snapshot.native_dataset,
    )


def test_fit_plot_weights_render_without_decoding_model(
    qtbot, fit_snapshot, monkeypatch
):
    def blocked_decode(_self):
        raise AssertionError("Native fit plotting must not decode its model")

    monkeypatch.setattr(FigureFitSourceSnapshot, "result_dataset", blocked_decode)
    tool = _tool(qtbot, fit_snapshot, FigureOperationState.fit_plot(source="edc_fit"))
    original = _render(tool)
    assert len(original.collections) == 1
    namespace = _exec_generated_code(
        tool.generated_code(), {"edc_fit": fit_snapshot.native_dataset}
    )
    _assert_equal_artists(original, namespace["fig"].axes[0])
    restored = ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    _assert_equal_artists(original, _render(restored))
    restored_snapshot = restored.source_payload("edc_fit", FigureFitSourceSnapshot)
    xr.testing.assert_identical(
        restored_snapshot.native_dataset.modelfit_weights,
        fit_snapshot.native_dataset.modelfit_weights,
    )


def test_fit_plot_editor_errorbar_styles_render_and_restore(qtbot, fit_snapshot):
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "fit_data_kw": {"markeredgecolor": "green", "markevery": [0, 2]},
            "fit_line_kw": {"markevery": (0, 2)},
        }
    )
    tool = _tool(qtbot, fit_snapshot, operation)
    editor = tool.operation_editor
    editor.select_section("fit_data")
    page = editor.current_page
    for suffix, value in (
        ("Color", "red"),
        ("Linewidth", "1.7"),
        ("Capsize", "4"),
        ("Capthick", "2.2"),
        ("Errorevery", "(1, 2)"),
    ):
        _set_text(page, "figureComposerFitPlotError" + suffix, value)
    _activate_combo(page, "figureComposerFitPlotErrorBarsabove", "above")
    _set_text(page, "figureComposerFitPlotDataColor", "black")
    kwargs = tool.tool_status.operations[0].fit_data_kw
    assert kwargs == {
        "markeredgecolor": "green",
        "markevery": [0, 2],
        "ecolor": "red",
        "elinewidth": 1.7,
        "capsize": 4.0,
        "capthick": 2.2,
        "errorevery": (1, 2),
        "barsabove": True,
        "color": "black",
    }
    original = _render(tool)
    errorbar = original.containers[0]
    data, caps, bars = errorbar.lines
    assert data.get_color() == "black"
    assert caps[0].get_markersize() == 8
    assert caps[0].get_markeredgewidth() == 2.2
    np.testing.assert_allclose(bars[0].get_colors(), [mcolors.to_rgba("red")])
    np.testing.assert_allclose(bars[0].get_linewidths(), [1.7])
    namespace = _exec_generated_code(
        tool.generated_code(), {"edc_fit": fit_snapshot.native_dataset}
    )
    _assert_equal_artists(original, namespace["fig"].axes[0])
    restored = ToolWindow.from_dataset(tool.to_dataset())
    qtbot.addWidget(restored)
    assert restored.tool_status.operations[0].fit_data_kw == kwargs
    assert restored.tool_status.operations[0].fit_line_kw == operation.fit_line_kw
    _assert_equal_artists(original, _render(restored))
    restored.operation_editor.select_section("fit_data")
    stride = restored.operation_editor.current_page.findChild(
        QtWidgets.QLineEdit, "figureComposerFitPlotErrorErrorevery"
    )
    assert stride is not None
    stride.editingFinished.emit()
    assert not restored.operation_editor.has_input_error(
        restored.tool_status.operations[0]
    )
    _assert_equal_artists(original, _render(restored))

    errors = page.findChild(QtWidgets.QCheckBox, "figureComposerFitPlotShowYerr")
    assert errors is not None
    errors.setChecked(False)
    assert tool.tool_status.operations[0].fit_data_kw == kwargs
    without_errors = _render(tool)
    assert not without_errors.collections
    namespace = _exec_generated_code(
        tool.generated_code(), {"edc_fit": fit_snapshot.native_dataset}
    )
    _assert_equal_artists(without_errors, namespace["fig"].axes[0])
    _set_text(page, "figureComposerFitPlotErrorCapsize", "0")
    assert tool.tool_status.operations[0].fit_data_kw["capsize"] == 0.0
    _set_text(page, "figureComposerFitPlotErrorCapsize", "")
    assert "capsize" not in tool.tool_status.operations[0].fit_data_kw
    _set_text(page, "figureComposerFitPlotErrorErrorevery", "0")
    assert tool.tool_status.operations[0].fit_data_kw["errorevery"] == (1, 2)
    assert editor.has_input_error(tool.tool_status.operations[0])
    _set_text(page, "figureComposerFitPlotErrorErrorevery", "")
    assert "errorevery" not in tool.tool_status.operations[0].fit_data_kw
    assert not editor.has_input_error(tool.tool_status.operations[0])


def test_fit_plot_editor_extra_properties_and_defaults(qtbot, fit_snapshot):
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={"fit_line_kw": {"c": "navy", "ls": "--", "rasterized": True}}
    )
    tool = _tool(qtbot, fit_snapshot, operation)
    editor = tool.operation_editor
    editor.select_section("fit_line")
    page = editor.current_page
    prefix = "figureComposerFitPlotBestFit"
    kwargs = page.findChild(QtWidgets.QLineEdit, prefix + "KwargsEdit")
    assert kwargs is not None
    assert kwargs.text() == "rasterized=True"
    _set_text(page, prefix + "Color", "red")
    _set_text(page, prefix + "Linewidth", "2.5")
    _activate_combo(page, prefix + "Linestyle", ":")
    assert tool.tool_status.operations[0].fit_line_kw == {
        "color": "red",
        "linestyle": ":",
        "linewidth": 2.5,
        "rasterized": True,
    }
    _set_kwargs(
        page, prefix, "rasterized=True, drawstyle='steps-mid', markevery=(0, 2)"
    )
    operation = tool.tool_status.operations[0]
    assert operation.fit_line_kw["drawstyle"] == "steps-mid"
    assert operation.fit_line_kw["markevery"] == (0, 2)
    assert _render(tool).lines[0].get_drawstyle() == "steps-mid"
    for text in (
        "markevery=[1,",
        "{'not a property': 1}",
        "linewidth=1",
        "lw=1",
        "color='green'",
        "c='green'",
        "x=[0, 1]",
        "yerr=0.1",
    ):
        _set_kwargs(page, prefix, text)
        assert tool.tool_status.operations[0].fit_line_kw == operation.fit_line_kw
        assert editor.has_input_error(tool.tool_status.operations[0])
    _set_kwargs(page, prefix, "{'markevery': 3}")
    assert tool.tool_status.operations[0].fit_line_kw == {
        "color": "red",
        "linestyle": ":",
        "linewidth": 2.5,
        "markevery": 3,
    }
    assert not editor.has_input_error(tool.tool_status.operations[0])
    _set_kwargs(page, prefix, "")
    assert tool.tool_status.operations[0].fit_line_kw == {
        "color": "red",
        "linestyle": ":",
        "linewidth": 2.5,
    }
    _set_text(page, prefix + "Linewidth", "")
    _set_text(page, prefix + "Color", "")
    assert tool.tool_status.operations[0].fit_line_kw == {"linestyle": ":"}


def test_fit_plot_editor_batch_styles_preserve_distinct_properties(qtbot, fit_snapshot):
    operations = tuple(
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={"fit_line_kw": kwargs}
        )
        for kwargs in (
            {"color": "red", "rasterized": True},
            {"color": "navy", "drawstyle": "steps-mid"},
        )
    )
    tool = FigureComposerTool.from_fit_sources(
        {"edc_fit": fit_snapshot},
        sources=(FigureSourceState(name="edc_fit", kind="fit"),),
        operations=operations,
    )
    qtbot.addWidget(tool)
    _select_operation_rows(tool, (0, 1))
    editor = tool.operation_editor
    editor.select_section("fit_line")
    page = editor.current_page
    prefix = "figureComposerFitPlotBestFit"
    color = page.findChild(QtWidgets.QLineEdit, prefix + "Color")
    assert color is not None
    assert color.property("batch_mixed")
    color.editingFinished.emit()
    kwargs = page.findChild(QtWidgets.QLineEdit, prefix + "KwargsEdit")
    assert kwargs is not None
    assert kwargs.property("batch_mixed")
    kwargs.editingFinished.emit()
    assert tool.tool_status.operations == operations
    _set_kwargs(page, prefix, "markevery=2")
    assert [operation.fit_line_kw for operation in tool.tool_status.operations] == [
        {"color": "red", "markevery": 2},
        {"color": "navy", "markevery": 2},
    ]
    _set_text(page, prefix + "Color", "black")
    assert [operation.fit_line_kw for operation in tool.tool_status.operations] == [
        {"color": "black", "markevery": 2},
        {"color": "black", "markevery": 2},
    ]


def test_fit_plot_editor_kwargs_preserve_errorbar_controls(qtbot, fit_snapshot):
    tool = _tool(
        qtbot,
        fit_snapshot,
        FigureOperationState.fit_plot(source="edc_fit").model_copy(
            update={"fit_data_kw": {"capsize": 3.0, "ecolor": "red"}}
        ),
    )
    editor = tool.operation_editor
    editor.select_section("fit_data")
    page = editor.current_page
    prefix = "figureComposerFitPlotData"
    _set_kwargs(page, prefix, "markeredgewidth=2")
    expected = {"capsize": 3.0, "ecolor": "red", "markeredgewidth": 2}
    assert tool.tool_status.operations[0].fit_data_kw == expected
    for text in ("capsize=5", "ecolor='green'", "errorevery=2", "xerr=0.1"):
        _set_kwargs(page, prefix, text)
        assert tool.tool_status.operations[0].fit_data_kw == expected
        assert editor.has_input_error(tool.tool_status.operations[0])
    _set_kwargs(page, prefix, "")
    assert tool.tool_status.operations[0].fit_data_kw == {
        "capsize": 3.0,
        "ecolor": "red",
    }
    assert not editor.has_input_error(tool.tool_status.operations[0])


def test_fit_components_editor_kwargs_preserve_other_styles(qtbot, component_snapshot):
    operation = FigureOperationState.fit_plot(source="edc_fit").model_copy(
        update={
            "fit_sigma": 1.0,
            "show_components": True,
            "fit_component_kw": {
                "peak_": {"color": "red", "rasterized": True},
                "bg_": {"color": "navy", "drawstyle": "steps-post"},
            },
            "fit_component_band_kw": {"peak_": {"alpha": 0.4}},
        }
    )
    tool = _tool(qtbot, component_snapshot, operation)
    editor = tool.operation_editor
    editor.select_section("fit_components")
    page = _component_page(editor, "peak_")
    _set_kwargs(page, "figureComposerFitPlotComponentLine", "drawstyle='steps-mid'")
    _set_kwargs(page, "figureComposerFitPlotComponentBand", "hatch='//'")
    updated = tool.tool_status.operations[0]
    assert updated.fit_component_kw == {
        "peak_": {"color": "red", "drawstyle": "steps-mid"},
        "bg_": {"color": "navy", "drawstyle": "steps-post"},
    }
    assert updated.fit_component_band_kw == {"peak_": {"alpha": 0.4, "hatch": "//"}}
    axis = _render(tool)
    assert axis.lines[-2].get_drawstyle() == "steps-mid"
    assert axis.collections[-2].get_hatch() == "//"
    _set_kwargs(page, "figureComposerFitPlotComponentBand", "alpha=0.2")
    assert tool.tool_status.operations[0].fit_component_band_kw == (
        updated.fit_component_band_kw
    )
    assert editor.has_input_error(tool.tool_status.operations[0])
    _set_kwargs(page, "figureComposerFitPlotComponentBand", "")
    assert tool.tool_status.operations[0].fit_component_band_kw == {
        "peak_": {"alpha": 0.4}
    }
    assert not editor.has_input_error(tool.tool_status.operations[0])


@pytest.mark.parametrize(
    ("section", "prefix", "weighted", "method"),
    [
        ("fit_data", "figureComposerFitPlotData", True, "errorbar"),
        ("fit_data", "figureComposerFitPlotData", False, "plot"),
        ("fit_line", "figureComposerFitPlotBestFit", True, "plot"),
        ("fit_band", "figureComposerFitPlotBand", True, "fill_between"),
        ("fit_components", "figureComposerFitPlotComponentLine", False, "plot"),
        (
            "fit_components",
            "figureComposerFitPlotComponentBand",
            False,
            "fill_between",
        ),
    ],
)
def test_fit_plot_editor_kwargs_documentation(
    qtbot,
    monkeypatch,
    fit_snapshot,
    component_snapshot,
    section,
    prefix,
    weighted,
    method,
):
    snapshot = fit_snapshot if weighted else component_snapshot
    tool = _tool(qtbot, snapshot, FigureOperationState.fit_plot(source="edc_fit"))
    editor = tool.operation_editor
    editor.select_section(section)
    page = (
        _component_page(editor, "peak_")
        if section == "fit_components"
        else editor.current_page
    )
    button = _style_control(page, QtWidgets.QToolButton, prefix + "KwargsDocsButton")
    assert button is not None
    expected = (
        f"https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.{method}.html"
    )
    assert button.property("figure_fit_style_doc_url") == expected
    opened = []
    monkeypatch.setattr(
        QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url.toString())
    )
    button.click()
    assert opened == [expected]


def test_fit_plot_array_source_selector_excludes_fit(qtbot, fit_snapshot):
    operation = FigureOperationState.line(label="Array profile", source="spectrum")
    tool = _tool(qtbot, fit_snapshot, operation)
    data = fit_snapshot.native_dataset.modelfit_data
    tool.add_sources((FigureSourceState(name="spectrum"),), {"spectrum": data})
    combo = tool.operation_editor.source_controls.findChild(
        QtWidgets.QComboBox, "figureComposerLineSourceCombo"
    )
    assert combo is not None
    assert combo.findData("edc_fit") == -1
    assert combo.findData("spectrum") >= 0
    xr.testing.assert_identical(tool.source_data()["spectrum"], data)


def test_fit_plot_editor_updates_sampling_sigma_and_visibility(qtbot, fit_snapshot):
    tool = _tool(qtbot, fit_snapshot, FigureOperationState.fit_plot(source="edc_fit"))
    editor = tool.operation_editor
    add = editor.source_controls.findChild(
        QtWidgets.QPushButton, "figureComposerFitPlotAddSourceButton"
    )
    assert add is not None
    assert not add.isEnabled()
    assert _render(tool).lines[0].get_xdata().size == 19
    editor.select_section("fit_line")
    sampling = editor.current_page.findChild(
        QtWidgets.QComboBox, "figureComposerFitPlotSampling"
    )
    assert sampling is not None
    assert sampling.isEnabled()
    sampling.setCurrentIndex(1)
    sampling.activated.emit(1)
    qtbot.waitUntil(
        lambda: editor.current_page.findChild(
            QtWidgets.QSpinBox, "figureComposerFitPlotPointCount"
        ).isEnabled()
    )
    count = editor.current_page.findChild(
        QtWidgets.QSpinBox, "figureComposerFitPlotPointCount"
    )
    assert count is not None
    assert count.isEnabled()
    count.setValue(71)
    assert tool.tool_status.operations[0].fit_numpoints == 71
    assert _render(tool).lines[0].get_xdata().size == 71

    editor.select_section("fit_band")
    band = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowConfidence"
    )
    assert band is not None
    assert band.isEnabled()
    band.setChecked(True)
    qtbot.waitUntil(
        lambda: editor.current_page.findChild(
            QtWidgets.QDoubleSpinBox, "figureComposerFitPlotSigmaSpin"
        ).isEnabled()
    )
    level = editor.current_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerFitPlotSigmaSpin"
    )
    assert level is not None
    assert level.value() == 1.0
    assert tool.tool_status.operations[0].fit_sigma == 1.0
    assert level.suffix() == ""
    widths = []
    for sigma in (1.0, 2.0, 3.0, 0.95):
        level.setValue(sigma)
        assert tool.tool_status.operations[0].fit_sigma == sigma
        vertices = _render(tool).collections[0].get_paths()[0].vertices
        widths.append(np.ptp(vertices[:, 1]))
    assert widths[0] < widths[1] < widths[2]
    assert widths[0] < widths[3] < widths[2]

    editor.select_section("fit_data")
    errors = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowYerr"
    )
    assert errors is not None
    assert errors.isEnabled()
    assert errors.isChecked()
    errors.setChecked(False)
    assert not tool.tool_status.operations[0].fit_show_yerr
    axis = _render(tool)
    assert len(axis.collections) == 1
    code = tool.generated_code()
    assert "show_yerr=False" in code
    namespace = _exec_generated_code(code, {"edc_fit": fit_snapshot.result_dataset()})
    _assert_equal_artists(axis, namespace["fig"].axes[0])
    data = editor.current_page.findChild(
        QtWidgets.QCheckBox, "figureComposerFitPlotShowData"
    )
    assert data is not None
    data.setChecked(False)
    assert not tool.tool_status.operations[0].show_data
    assert len(_render(tool).lines) == 1
    editor.select_section("fit_line")
    fit = editor.current_page.findChild(QtWidgets.QCheckBox)
    assert fit is not None
    fit.setChecked(False)
    assert not tool.tool_status.operations[0].show_fit
    axis = _render(tool)
    assert not axis.lines
    assert len(axis.collections) == 1


@pytest.mark.parametrize("accepted", [False, True])
def test_fit_plot_add_source_button_requests_source(qtbot, fit_snapshot, accepted):
    tool = _tool(qtbot, fit_snapshot, FigureOperationState.fit_plot(source="edc_fit"))
    requests = []

    def add_source(kinds):
        assert kinds == ("fit",)
        requests.append(True)
        if accepted:
            tool.add_fit_sources(
                (FigureSourceState(name="other_fit", kind="fit"),),
                {"other_fit": fit_snapshot},
            )
        return accepted

    tool._set_source_add_callbacks(add_sources=add_source)
    add = next(
        button
        for button in tool.operation_editor.source_controls.findChildren(
            QtWidgets.QPushButton, "figureComposerFitPlotAddSourceButton"
        )
        if button.isEnabled()
    )
    add.click()
    assert requests == [True]
    assert tool.tool_status.operations[0].sources == (
        "other_fit" if accepted else "edc_fit",
    )
    assert tool.source_kind_names("fit") == (
        ("edc_fit", "other_fit") if accepted else ("edc_fit",)
    )


def test_fit_plot_source_dialog_preserves_active_sender(qtbot, fit_snapshot):
    tool = _tool(qtbot, fit_snapshot, FigureOperationState.fit_plot(source="edc_fit"))
    editor = tool.operation_editor
    dialog = QtWidgets.QDialog(tool)
    qtbot.addWidget(dialog)
    observations = []

    def inspect_sender():
        observations.append(
            (
                qt_is_valid(add),
                editor._generation == generation,
                editor.active_signal_widget is add,
            )
        )
        dialog.reject()

    def add_source(kinds):
        assert kinds == ("fit",)
        tool._queue_operation_editor_update()
        # Source changes can retire a control before its callback opens a dialog.
        editor._retire_widget(add)
        QtCore.QTimer.singleShot(350, inspect_sender)
        dialog.exec()
        return False

    tool._set_source_add_callbacks(add_sources=add_source)
    add = next(
        button
        for button in editor.source_controls.findChildren(
            QtWidgets.QPushButton, "figureComposerFitPlotAddSourceButton"
        )
        if button.isEnabled()
    )
    generation = editor._generation
    add.click()
    assert observations == [(True, True, True)]
    qtbot.waitUntil(lambda: not tool._operation_editor_update_pending, timeout=1000)
    qtbot.waitUntil(lambda: not qt_is_valid(add), timeout=1000)


def test_fit_source_available_to_python_step(qtbot, fit_snapshot):
    operation = FigureOperationState.custom(
        label="Fit result in Python",
        code="eplt.plot_fit(edc_fit, ax=ax, numpoints=37, sigma=0.9)",
    )
    tool = _tool(qtbot, fit_snapshot, operation)
    rendered = _render(tool)
    namespace = _exec_generated_code(
        tool.generated_code(), {"edc_fit": fit_snapshot.result_dataset()}
    )
    _assert_equal_artists(rendered, namespace["fig"].axes[0])
    assert rendered.lines[0].get_xdata().size == 37
    assert tool._document.source_usage_count("edc_fit") == 1
