from unittest.mock import patch

import numpy as np
import pydantic
import pytest
import xarray as xr
from matplotlib.figure import Figure
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureOperationKind,
    FigureOperationState,
    FigureRecipeState,
    FigureSubplotsState,
)
from erlab.interactive.imagetool._provenance._model import ScriptInput
from erlab.interactive.imagetool.manager._figurecomposer import _dialogs
from tests.interactive.imagetool.manager.helpers import select_child_tool, select_tools

from .test_manager_sources import _source_picker_item


class _ProviderToolState(pydantic.BaseModel):
    scale: float = 1.0


class _ArrayProvider:
    payload_kind = "array"

    def __init__(self, tool, provider_id):
        self._tool = tool
        self.provider_id = provider_id
        self.label = provider_id

    def is_ready(self, selection=None):
        return self._tool.ready

    def capture(self, selection=None):
        from erlab.interactive._figurecomposer._providers import FigureSourceCapture
        from erlab.interactive._figurecomposer._sources import ArraySourcePayload

        selection = (
            {"scale": self._tool.tool_status.scale} if selection is None else selection
        )
        return FigureSourceCapture(
            payload=ArraySourcePayload(
                (self._tool.tool_data * selection["scale"]).copy(deep=True)
            ),
            name=self.provider_id,
            selection=dict(selection),
            revision_token=str(self._tool.provenance_revision),
        )


class _ProviderTool(erlab.interactive.utils.ToolWindow[_ProviderToolState]):
    StateModel = _ProviderToolState
    tool_name = "provider-test"

    def __init__(self, data):
        super().__init__()
        self._data = data
        self._status = _ProviderToolState()
        self.ready = True

    @property
    def tool_data(self):
        return self._data

    @property
    def tool_status(self):
        return self._status

    @tool_status.setter
    def tool_status(self, state):
        self._status = state

    def figure_source_providers(self):
        return (_ArrayProvider(self, "first"), _ArrayProvider(self, "second"))


def test_manager_discovers_and_refreshes_multiple_nonfit_outputs(
    qtbot, manager_context
):
    from erlab.interactive._figurecomposer._providers import FigureSourceTarget

    data = xr.DataArray(np.arange(4.0), dims="x", coords={"x": np.arange(4.0)})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        producer = _ProviderTool(data)
        producer.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        producer_uid = manager.add_childtool(
            producer, script_inputs={"data": 0}, show=False
        )
        assert (
            erlab.interactive.utils.ToolWindow.figure_source_providers(producer) == ()
        )
        workflow = manager._figure_workflows
        assert workflow._figure_node_is_source(manager._node_for_target(producer_uid))
        picker = _dialogs._FigureSourcePickerDialog(manager, source_kinds=("array",))
        qtbot.addWidget(picker)
        first = FigureSourceTarget(producer_uid, "first")
        second = FigureSourceTarget(producer_uid, "second")
        items = [
            item
            for item in picker._tree_items()
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) in (first, second)
        ]
        assert len(items) == 2
        items[0].setCheckState(0, QtCore.Qt.CheckState.Checked)
        assert picker.selected_targets() == (first,)
        filtered_picker = _dialogs._FigureSourcePickerDialog(
            manager, source_kinds=("fit",), prechecked_uids=(producer_uid,)
        )
        qtbot.addWidget(filtered_picker)
        assert not filtered_picker.selected_targets()

        figure_uid = workflow.create_figure_from_targets(
            picker.selected_targets(), show=False
        )
        composer = manager._child_node(figure_uid).tool_window
        [source] = composer.source_states()
        assert source.kind == "array"
        assert source.provider_id == "first"
        assert source.provider_selection == {"scale": 1.0}
        assert not source.use_node_data
        assert source.to_script_input() is None
        assert len(composer.tool_status.operations) == 1
        assert workflow._can_reveal_figure_source(figure_uid, source.name)
        producer._data = data + 2.0
        producer.tool_status = _ProviderToolState(scale=3.0)
        xr.testing.assert_identical(composer.source_data()[source.name], data)
        assert workflow._can_refresh_figure_source(figure_uid, source.name)
        assert workflow._refresh_figure_source(figure_uid, source.name)
        xr.testing.assert_identical(composer.source_data()[source.name], data + 2.0)
        assert composer.source_states()[0].provider_selection == {"scale": 1.0}

        assert workflow._add_imagetool_sources_to_figure(figure_uid, (second,))
        assert {item.provider_id for item in composer.source_states()} == {
            "first",
            "second",
        }
        assert len(composer.tool_status.operations) == 1
        second_source = next(
            item for item in composer.source_states() if item.provider_id == "second"
        )
        xr.testing.assert_identical(
            composer.source_data()[second_source.name], (data + 2.0) * 3.0
        )
        assert second_source.to_script_input() is None
        assert not second_source.use_node_data
        with composer._save_tool_data_reference_context():
            assert composer._source_reference_payload(second_source.name) is None
        assert workflow.append_figure_from_targets(
            (second,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )
        assert len(composer.tool_status.operations) == 2
        assert len(composer.source_states()) == 2
        all_uid = workflow.create_figure_from_targets((producer_uid,), show=False)
        all_composer = manager._child_node(all_uid).tool_window
        assert len(all_composer.source_states()) == 2
        assert len(all_composer.tool_status.operations) == 2
        producer.ready = False
        assert not workflow._can_refresh_figure_source(figure_uid, source.name)
        assert not workflow._refresh_figure_source(figure_uid, source.name)
        assert not workflow._figure_node_is_source(
            manager._node_for_target(producer_uid)
        )
        unavailable_picker = _dialogs._FigureSourcePickerDialog(
            manager, prechecked_uids=(producer_uid,)
        )
        qtbot.addWidget(unavailable_picker)
        unavailable = [
            item
            for item in unavailable_picker._tree_items()
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) in (first, second)
        ]
        assert len(unavailable) == 2
        assert all(
            not item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable
            for item in unavailable
        )
        assert unavailable_picker.selected_targets() == ()
        assert not unavailable_picker.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).isEnabled()


def test_source_picker_filters_individual_outputs_and_drops_removed_tools(
    qtbot, manager_context, monkeypatch
):
    from erlab.interactive._figurecomposer._providers import FigureSourceTarget

    data = xr.DataArray(np.arange(4.0), dims="x", coords={"x": np.arange(4.0)})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        producer = _ProviderTool(data)
        producer.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        producer_uid = manager.add_childtool(
            producer, script_inputs={"data": 0}, show=False
        )
        first, second = producer.figure_source_providers()
        second.payload_kind = "fit"
        monkeypatch.setattr(
            producer, "figure_source_providers", lambda: (first, second)
        )
        picker = _dialogs._FigureSourcePickerDialog(
            manager, source_kinds=("array",), prechecked_uids=(producer_uid,)
        )
        qtbot.addWidget(picker)
        selected = FigureSourceTarget(producer_uid, "first")
        assert picker.selected_targets() == (selected,)
        workflow = manager._figure_workflows
        figure_uid = workflow.create_figure_from_targets(
            picker.selected_targets(), show=False
        )
        composer = manager._child_node(figure_uid).tool_window
        [source] = composer.source_states()
        assert source.provider_id == "first"
        snapshot = composer.source_payloads()[source.name].data
        manager._remove_childtool(producer_uid)
        assert picker.selected_targets() == ()
        assert not workflow._can_refresh_figure_source(figure_uid, source.name)
        assert not workflow._refresh_figure_source(figure_uid, source.name)
        assert not workflow._can_reveal_figure_source(figure_uid, source.name)
        xr.testing.assert_identical(
            composer.source_payloads()[source.name].data, snapshot
        )


def test_provider_capture_deduplicates_outputs_and_rejects_incompatible_payload(
    qtbot, manager_context, monkeypatch
):
    from erlab.interactive._figurecomposer._providers import FigureSourceTarget

    data = xr.DataArray(np.arange(4.0), dims="x", coords={"x": np.arange(4.0)})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        producer = _ProviderTool(data)
        producer.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        producer_uid = manager.add_childtool(
            producer, script_inputs={"data": 0}, show=False
        )
        first = FigureSourceTarget(producer_uid, "first")
        workflow = manager._figure_workflows
        figure_uid = workflow.create_figure_from_targets(
            (producer_uid, first), show=False
        )
        composer = manager._child_node(figure_uid).tool_window
        assert [source.provider_id for source in composer.source_states()] == [
            "first",
            "second",
        ]
        assert len(composer.tool_status.operations) == 2
        initial = composer.tool_status
        snapshots = composer.source_payloads()
        [provider, _] = producer.figure_source_providers()
        provider.payload_kind = "fit"
        monkeypatch.setattr(producer, "figure_source_providers", lambda: (provider,))
        assert not workflow._can_refresh_figure_source(
            figure_uid, composer.source_states()[0].name
        )
        assert not workflow._refresh_figure_source(
            figure_uid, composer.source_states()[0].name
        )
        with pytest.raises(ValueError, match="incompatible payload"):
            workflow.create_figure_from_targets((first,), show=False)
        with pytest.raises(ValueError, match="incompatible payload"):
            workflow.append_figure_from_targets(
                (first,),
                figure_uid=figure_uid,
                axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
                show=False,
            )
        assert tuple(manager._figure_uids()) == (figure_uid,)
        assert composer.tool_status == initial
        for name, payload in snapshots.items():
            xr.testing.assert_identical(
                composer.source_payloads()[name].data, payload.data
            )


@pytest.mark.parametrize("action", ["create", "append"])
def test_provider_default_plots_retain_momentum_overlay(qtbot, manager_context, action):
    from erlab.interactive._figurecomposer import FigureOperationKind
    from erlab.interactive._figurecomposer._providers import FigureSourceTarget

    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("ky", "kx"),
        coords={"ky": np.linspace(-1, 1, 3), "kx": np.linspace(-2, 2, 4)},
    )
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        producer = _ProviderTool(data)
        producer.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        producer_uid = manager.add_childtool(
            producer, script_inputs={"data": 0}, show=False
        )
        targets = (FigureSourceTarget(producer_uid, "first"),)
        if action == "create":
            figure_uid = manager._figure_workflows.create_figure_from_targets(
                targets, show=False
            )
        else:
            figure_uid = manager.create_empty_figure(show=False)
            assert manager._figure_workflows.append_figure_from_targets(
                targets,
                figure_uid=figure_uid,
                axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
                show=False,
            )
        composer = manager._child_node(figure_uid).tool_window
        assert [operation.kind for operation in composer.tool_status.operations] == [
            FigureOperationKind.PLOT_ARRAY,
            FigureOperationKind.BZ_OVERLAY,
        ]
        assert composer.tool_status.operations[-1].bz_bounds == (-2.0, 2.0, -1.0, 1.0)


@pytest.mark.parametrize("action", ["create", "append", "prompt"])
def test_provider_plot_selection_errors_are_reported(
    qtbot, manager_context, monkeypatch, action
):
    from erlab.interactive._figurecomposer._exceptions import (
        FigureComposerPlotSlicesSelectionError,
    )
    from erlab.interactive._figurecomposer._providers import FigureSourceTarget

    data = xr.DataArray(np.arange(4.0), dims="x", coords={"x": np.arange(4.0)})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        producer = _ProviderTool(data)
        producer.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        producer_uid = manager.add_childtool(
            producer, script_inputs={"data": 0}, show=False
        )
        workflow = manager._figure_workflows
        error = FigureComposerPlotSlicesSelectionError()

        def raise_selection_error(*args, **kwargs):
            raise error

        reports = []
        monkeypatch.setattr(
            workflow, "_source_operations_for_payloads", raise_selection_error
        )
        monkeypatch.setattr(
            workflow, "_show_figure_plot_slices_selection_error", reports.append
        )
        targets = (FigureSourceTarget(producer_uid, "first"),)
        if action == "create":
            assert workflow.create_figure_from_targets(targets, show=False) is None
            assert not manager._figure_uids()
        else:
            figure_uid = manager.create_empty_figure(show=False)
            assert not workflow.append_figure_from_targets(
                targets,
                figure_uid=figure_uid,
                axes_selection=None
                if action == "prompt"
                else FigureAxesSelectionState(axes=((0, 0),)),
                show=False,
            )
            composer = manager._child_node(figure_uid).tool_window
            assert not composer.source_states()
            assert not composer.tool_status.operations
        assert reports == [error]


@pytest.mark.parametrize(
    "operation", [None, FigureOperationState.fit_plot(source="fit")]
)
def test_fit_target_dialog_requires_one_axis(qtbot, manager_context, operation):
    with manager_context() as manager:
        composer = FigureComposerTool(
            xr.DataArray(np.empty(0, dtype=np.uint8), dims="empty"),
            recipe=FigureRecipeState(setup=FigureSubplotsState(nrows=1, ncols=2)),
            source_data={},
        )
        figure_uid = manager.add_figuretool(composer, show=False)
        dialog = _dialogs._AppendFigureTargetDialog(
            manager, (figure_uid,), operation, source_kind="fit"
        )
        qtbot.addWidget(dialog)
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 0),))
        assert dialog._ok_button().isEnabled()
        dialog._select_all_axes()
        assert not dialog._ok_button().isEnabled()
        assert dialog.selected_target() is None
        dialog.axes_selector.set_selected_axes(((0, 1),), emit=True)
        assert dialog._ok_button().isEnabled()
        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes=((0, 1),)),
        )


def _two_axis_setup(layout_mode):
    if layout_mode == "subplots":
        return FigureSubplotsState(nrows=1, ncols=2)
    return FigureSubplotsState(
        layout_mode="gridspec",
        gridspec=FigureGridSpecLayoutState(
            root=FigureGridSpecGridState(
                grid_id="root",
                nrows=1,
                ncols=2,
                axes=tuple(
                    FigureGridSpecAxesState(
                        axes_id=f"axis-{col}",
                        span=FigureGridSpecSpanState(
                            row_start=0,
                            row_stop=1,
                            col_start=col,
                            col_stop=col + 1,
                        ),
                    )
                    for col in range(2)
                ),
            )
        ),
    )


@pytest.mark.parametrize("layout_mode", ["subplots", "gridspec"])
@pytest.mark.parametrize("explicit", [False, True])
def test_mixed_default_operation_targets(qtbot, manager_context, layout_mode, explicit):
    with manager_context() as manager:
        composer = FigureComposerTool(
            xr.DataArray(np.empty(0, dtype=np.uint8), dims="empty"),
            recipe=FigureRecipeState(setup=_two_axis_setup(layout_mode)),
            source_data={},
        )
        figure_uid = manager.add_figuretool(composer, show=False)
        line = FigureOperationState.line(label="line", source="data")
        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            line if explicit else None,
            allow_new_figure=True,
            source_count=2,
            source_kind=None,
            default_operations=(line, FigureOperationState.fit_plot(source="fit")),
        )
        qtbot.addWidget(dialog)
        expected = (
            FigureAxesSelectionState(axes=((0, 0),))
            if layout_mode == "subplots"
            else FigureAxesSelectionState(axes_ids=("axis-0",))
        )
        assert dialog.axes_selection() == expected
        assert dialog._ok_button().isEnabled()
        dialog._select_all_axes()
        assert dialog._ok_button().isEnabled() is explicit
        assert (dialog.selected_target() is not None) is explicit
        for action in (_dialogs._FIGURE_DIALOG_NEW, _dialogs._FIGURE_DIALOG_ADD_SOURCE):
            dialog.action_combo.setCurrentIndex(dialog.action_combo.findData(action))
            assert dialog._ok_button().isEnabled()
            assert dialog.selected_target() is None
        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_REPLACE_SOURCE)
        )
        assert not dialog._ok_button().isEnabled()


@pytest.mark.parametrize("layout_mode", ["subplots", "gridspec"])
@pytest.mark.parametrize("workflow", ["selection", "append"])
@pytest.mark.parametrize("accepted", [False, True])
def test_manager_mixed_source_append_targets(
    qtbot, manager_context, monkeypatch, layout_mode, workflow, accepted
):
    from erlab.analysis.fit.models import PolynomialModel
    from erlab.interactive._figurecomposer._rendering import _render_into_figure

    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(2 * x + 3, dims="x", coords={"x": x})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        fit_tool = erlab.interactive.ftool(
            data, model=PolynomialModel(degree=1), execute=False
        )
        fit_tool.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        fit_uid = manager.add_childtool(fit_tool, script_inputs={"data": 0}, show=False)
        result = data.xlm.modelfit("x", model=fit_tool._model, params=fit_tool._params)
        fit_tool._set_fit_ds(result, 0.0)
        composer = FigureComposerTool(
            xr.DataArray(np.empty(0, dtype=np.uint8), dims="empty"),
            recipe=FigureRecipeState(setup=_two_axis_setup(layout_mode)),
            source_data={},
        )
        figure_uid = manager.add_figuretool(composer, show=False)
        before = composer.tool_status
        figure_uids = tuple(manager._figure_uids())
        selected = (
            FigureAxesSelectionState(axes=((0, 1),))
            if layout_mode == "subplots"
            else FigureAxesSelectionState(axes_ids=("axis-1",))
        )
        prompted = []

        def exec_target_dialog(dialog):
            prompted.append(dialog)
            assert dialog._ok_button().isEnabled()
            assert dialog.axes_selection() != selected
            dialog._select_all_axes()
            assert not dialog._ok_button().isEnabled()
            assert dialog.selected_target() is None
            if layout_mode == "subplots":
                dialog.axes_selector.set_selected_axes(selected.axes, emit=True)
            else:
                dialog.gridspec_axes_selector.set_selected_axes_ids(
                    selected.axes_ids, emit=True
                )
            assert dialog._ok_button().isEnabled()
            assert dialog.selected_target() == (figure_uid, selected)
            return (
                QtWidgets.QDialog.DialogCode.Accepted
                if accepted
                else QtWidgets.QDialog.DialogCode.Rejected
            )

        monkeypatch.setattr(
            _dialogs._AppendFigureTargetDialog, "exec", exec_target_dialog
        )
        if workflow == "selection":
            manager.tree_view.clearSelection()
            select_tools(manager, [0])
            select_child_tool(manager, fit_uid)
            manager.create_figure_from_selection()
        else:
            assert (
                manager.append_figure_from_targets(
                    (0, fit_uid), figure_uid=figure_uid, show=False
                )
                is accepted
            )
        assert len(prompted) == 1
        assert tuple(manager._figure_uids()) == figure_uids
        if not accepted:
            assert composer.tool_status == before
            assert not composer.source_states()
            return
        assert {source.kind for source in composer.source_states()} == {"array", "fit"}
        assert [operation.kind for operation in composer.tool_status.operations] == [
            FigureOperationKind.LINE,
            FigureOperationKind.FIT_PLOT,
        ]
        assert all(
            operation.axes == selected for operation in composer.tool_status.operations
        )
        _render_into_figure(composer, composer.figure, sync_visible=False)
        assert not composer._operation_render_errors
        assert len(composer.figure.axes[1].lines) == 3


@pytest.mark.parametrize("action", ["create", "append", "source"])
def test_manager_fit_workflows_skip_unavailable_targets(qtbot, manager_context, action):
    from erlab.analysis.fit.models import PolynomialModel
    from erlab.interactive._figurecomposer import FigureAxesSelectionState

    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(2 * x + 3, dims="x", coords={"x": x})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        fit_tool = erlab.interactive.ftool(
            data, model=PolynomialModel(degree=1), execute=False
        )
        fit_tool.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        fit_uid = manager.add_childtool(fit_tool, script_inputs={"data": 0}, show=False)
        result = data.xlm.modelfit("x", model=fit_tool._model, params=fit_tool._params)
        fit_tool._set_fit_ds(result, 0.0)
        targets = ("unavailable-fit", fit_uid)
        if action == "create":
            figure_uid = manager._figure_workflows.create_figure_from_targets(
                targets, title="Fit figure", show=False
            )
            assert figure_uid is not None
        else:
            figure_uid = manager.create_empty_figure(show=False)
            if action == "append":
                composer = manager._child_node(figure_uid).tool_window
                assert composer._grow_subplot_grid("column")
                before = composer.tool_status
                assert not manager._figure_workflows.append_figure_from_targets(
                    (0, fit_uid),
                    figure_uid=figure_uid,
                    axes_selection=FigureAxesSelectionState(axes=((0, 0), (0, 1))),
                    show=False,
                )
                assert composer.tool_status == before
                assert not composer.source_states()
                assert manager._figure_workflows.append_figure_from_targets(
                    targets,
                    figure_uid=figure_uid,
                    axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
                    show=False,
                )
            else:
                assert manager._figure_workflows._add_fit_targets_to_figure(
                    figure_uid, targets
                )
        composer = manager._child_node(figure_uid).tool_window
        assert isinstance(composer, FigureComposerTool)
        assert len(composer.source_states()) == 1
        assert composer.source_states()[0].node_uid == fit_uid
        assert len(composer.tool_status.operations) == (action != "source")
        if action == "create":
            assert composer._tool_display_name == "Fit figure"


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("argument", ["fit_plot", "custom_code", "custom_operation"])
def test_manager_explicit_fit_requests_are_preserved(
    qtbot, manager_context, mixed, argument
):
    from erlab.analysis.fit.models import PolynomialModel
    from erlab.interactive._figurecomposer._rendering import _render_into_figure

    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(2 * x + 3 + 0.01 * np.cos(5 * x), dims="x", coords={"x": x})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        fit_tool = erlab.interactive.ftool(
            data, model=PolynomialModel(degree=1), execute=False
        )
        fit_tool.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        fit_uid = manager.add_childtool(fit_tool, script_inputs={"data": 0}, show=False)
        result = data.xlm.modelfit("x", model=fit_tool._model, params=fit_tool._params)
        fit_tool._set_fit_ds(result, 0.0)
        alias = manager._script_input_name_for_node(manager._node_for_target(fit_uid))
        code = f"fig._fit_total = float({alias}.modelfit_best_fit.sum())"
        requested = FigureOperationState.fit_plot(source=alias).model_copy(
            update={
                "fit_numpoints": 101,
                "fit_sigma": 0.95,
                "show_data": False,
                "fit_line_kw": {"color": "red"},
            }
        )
        kwargs = (
            {"operation": requested}
            if argument == "fit_plot"
            else {"custom_code": code}
            if argument == "custom_code"
            else {"operation": FigureOperationState.custom(label="summary", code=code)}
        )
        targets = (0, fit_uid) if mixed else (fit_uid,)
        figure_uid = manager.create_figure_from_targets(targets, show=False, **kwargs)
        assert figure_uid is not None
        composer = manager._child_node(figure_uid).tool_window
        assert isinstance(composer, FigureComposerTool)
        assert len(composer.source_states()) == 1 + mixed
        [stored] = composer.tool_status.operations
        _render_into_figure(composer, composer.figure, sync_visible=False)
        assert not composer._operation_render_errors
        if argument == "fit_plot":
            assert stored.fit_numpoints == requested.fit_numpoints
            assert stored.fit_sigma == requested.fit_sigma
            assert stored.show_data is False
            assert stored.fit_line_kw == requested.fit_line_kw
            [line] = composer.figure.axes[0].lines
            assert line.get_color() == "red"
            assert len(line.get_xdata()) == 101
            assert len(composer.figure.axes[0].collections) == 1
            if not mixed:
                append_uid = manager.create_empty_figure(show=False)
                assert manager.append_figure_from_targets(
                    targets,
                    figure_uid=append_uid,
                    axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
                    operation=requested,
                    show=False,
                )
                appended = manager._child_node(append_uid).tool_window
                assert appended.tool_status.operations[0].fit_numpoints == 101
                assert appended.tool_status.operations[0].fit_line_kw == {
                    "color": "red"
                }
        else:
            assert composer.figure._fit_total == pytest.approx(
                float(result.modelfit_best_fit.sum())
            )
        before = tuple(manager._figure_uids())
        for incompatible in (
            FigureOperationState.plot_array(label="array", source=alias),
            FigureOperationState.line(label="line", source=alias),
        ):
            with pytest.raises(ValueError, match="incompatible"):
                manager.create_figure_from_targets(
                    targets, operation=incompatible, show=False
                )
        assert tuple(manager._figure_uids()) == before


@pytest.mark.parametrize("first_kind", ["array", "fit"])
def test_explicit_fit_creation_preserves_existing_source_aliases(
    qtbot, manager_context, first_kind
):
    from erlab.analysis.fit.models import PolynomialModel
    from erlab.interactive._figurecomposer._rendering import _render_into_figure

    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(2 * x + 3, dims="x", coords={"x": x}, name="data_fit")
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        fit_uids = []
        for offset in (0.0, 1.0):
            fit_tool = erlab.interactive.ftool(
                data + offset, model=PolynomialModel(degree=1), execute=False
            )
            fit_tool.set_script_inputs(
                (ScriptInput(name="data"),), primary_input="data"
            )
            fit_uid = manager.add_childtool(
                fit_tool, script_inputs={"data": 0}, show=False
            )
            result = (data + offset).xlm.modelfit(
                "x", model=fit_tool._model, params=fit_tool._params
            )
            fit_tool._set_fit_ds(result, 0.0)
            fit_uids.append(fit_uid)
        targets = (0, fit_uids[1]) if first_kind == "array" else tuple(fit_uids)
        kwargs = (
            {"custom_code": "fig._source_total = float(data_fit.sum())"}
            if first_kind == "array"
            else {"operation": FigureOperationState.fit_plot(source="data_fit")}
        )
        figure_uid = manager.create_figure_from_targets(targets, show=False, **kwargs)
        composer = manager._child_node(figure_uid).tool_window
        assert tuple(source.name for source in composer.source_states()) == (
            "data_fit",
            "data_fit_2",
        )
        _render_into_figure(composer, composer.figure, sync_visible=False)
        assert not composer._operation_render_errors
        if first_kind == "array":
            assert composer.figure._source_total == float(data.sum())
        else:
            assert composer.tool_status.operations[0].sources == ("data_fit",)
            assert composer.source_states()[0].node_uid == fit_uids[0]
            np.testing.assert_allclose(
                composer.figure.axes[0].lines[1].get_ydata(), data.values
            )


@pytest.mark.parametrize(
    "action",
    [
        _dialogs._FIGURE_DIALOG_NEW,
        _dialogs._FIGURE_DIALOG_ADD_STEP,
        _dialogs._FIGURE_DIALOG_ADD_SOURCE,
        _dialogs._FIGURE_DIALOG_REPLACE_SOURCE,
        "cancel",
    ],
)
def test_fit_action_destination_preserves_snapshot_and_recipe(
    qtbot, manager_context, accept_dialog, action
):
    from erlab.analysis.fit.models import PolynomialModel

    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(2 * x + 3 + 0.01 * np.cos(5 * x), dims="x", coords={"x": x})
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        fit_tools = []
        fit_uids = []
        for offset in (0.0, 1.0):
            tool = erlab.interactive.ftool(
                data + offset, model=PolynomialModel(degree=1), execute=False
            )
            tool.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
            fit_uids.append(
                manager.add_childtool(tool, script_inputs={"data": 0}, show=False)
            )
            fitted = (data + offset).xlm.modelfit(
                "x", model=tool._model, params=tool._params
            )
            tool._set_fit_ds(fitted, 0.0)
            fit_tools.append(tool)
        workflow = manager._figure_workflows
        figure_uid = manager.create_figure_from_targets((fit_uids[0],), show=False)
        composer = manager._child_node(figure_uid).tool_window
        initial = composer.tool_status
        [original] = composer.source_states()
        original_snapshot = composer.source_payloads()[original.name].native_dataset
        select_child_tool(manager, figure_uid)

        def choose_destination(dialog):
            assert isinstance(dialog, _dialogs._AppendFigureTargetDialog)
            assert dialog.figure_uid() == figure_uid
            assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_STEP
            if action != "cancel":
                dialog.action_combo.setCurrentIndex(
                    dialog.action_combo.findData(action)
                )
            if action == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE:
                assert dialog.selected_source_alias() == original.name
            assert dialog._ok_button().isEnabled()

        returned = []
        accept_dialog(
            lambda: returned.append(workflow.add_fit_to_figure(fit_tools[1])),
            pre_call=choose_destination,
            accept_call=(lambda dialog: dialog.reject())
            if action == "cancel"
            else None,
        )
        if action == "cancel":
            assert returned == [None]
            assert tuple(manager._figure_uids()) == (figure_uid,)
            assert composer.tool_status == initial
        else:
            target_uid = returned[0]
            assert target_uid is not None
            target = manager._child_node(target_uid).tool_window
            source = next(
                source
                for source in target.source_states()
                if source.node_uid == fit_uids[1]
            )
            xr.testing.assert_allclose(
                target.source_payloads()[source.name].native_dataset.modelfit_best_fit,
                fitted.modelfit_best_fit,
            )
            assert workflow._can_refresh_figure_source(target_uid, source.name)
            if action == _dialogs._FIGURE_DIALOG_NEW:
                assert target_uid != figure_uid
                assert composer.tool_status == initial
                assert len(target.tool_status.operations) == 1
            else:
                assert target_uid == figure_uid
                if action == _dialogs._FIGURE_DIALOG_ADD_STEP:
                    assert target.tool_status.operations[0] == initial.operations[0]
                    assert target.tool_status.operations[-1].sources == (source.name,)
                    assert len(target.source_states()) == 2
                else:
                    assert target.tool_status.operations == initial.operations
                    assert len(target.source_states()) == (
                        1 if action == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE else 2
                    )
                if action == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE:
                    assert source.name == original.name
        if action != _dialogs._FIGURE_DIALOG_REPLACE_SOURCE:
            xr.testing.assert_identical(
                composer.source_payloads()[original.name].native_dataset,
                original_snapshot,
            )


def test_manager_fit_sources_replay_refresh_and_retain_snapshot(
    qtbot, manager_context, monkeypatch
):
    x = np.linspace(-1.0, 1.0, 41)
    data = xr.DataArray(
        np.stack([np.exp(-(x**2)) + 0.01 * np.sin(9 * x), 1.1 * np.exp(-(x**2))]),
        dims=("cut", "energy"),
        coords={"cut": [10.0, 20.0], "energy": x},
        name="spectra",
    )
    with manager_context() as manager:
        erlab.interactive.itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1)
        fit_tool = erlab.interactive.ftool(data, execute=False)
        fit_tool.set_script_inputs((ScriptInput(name="data"),), primary_input="data")
        fit_uid = manager.add_childtool(
            fit_tool,
            script_inputs={item.name: 0 for item in fit_tool.script_inputs},
            show=False,
        )
        fit_tool.domain_min_spin.setValue(-0.8)
        fit_tool.domain_max_spin.setValue(0.8)
        fit_tool.normalize_check.setChecked(True)
        job = fit_tool._fit_job()
        fitted = (
            fit_tool._fit_data()
            .xlm.modelfit(
                "energy",
                model=fit_tool._model,
                params=fit_tool._params,
                method=fit_tool.method_combo.currentText(),
                scale_covar=fit_tool.scale_covar_check.isChecked(),
            )
            .load()
        )
        fit_tool._set_fit_ds(fit_tool._fit_result_for_job(fitted, job), 0.0)
        pinned = fit_tool.figure_fit_selection()
        fit_tool.add_to_figure_action.trigger()
        figure_uid = manager._figure_uids()[0]
        assert figure_uid is not None
        composer = manager._child_node(figure_uid).tool_window
        assert isinstance(composer, FigureComposerTool)
        source = composer.source_states()[0]
        assert source.kind == "fit"
        assert source.provider_selection == pinned
        assert source.node_uid == fit_uid
        composer.add_operation(
            FigureOperationState.fit_plot(source=source.name).model_copy(
                update={"fit_sigma": 0.95}
            )
        )
        provenance = composer.current_provenance_spec()
        assert provenance is not None
        code = provenance.display_code()
        namespace = {"data": data}
        with patch(
            "erlab.plotting.plot_fit", wraps=erlab.plotting.plot_fit
        ) as plot_fit:
            exec(code, namespace)  # noqa: S102
        assert plot_fit.call_count == 2
        replayed = plot_fit.call_args_list[0].args[0]
        assert plot_fit.call_args_list[1].args[0] is replayed
        assert replayed.modelfit_results.size == 1
        xr.testing.assert_allclose(replayed.modelfit_data, fitted.modelfit_data)
        np.testing.assert_allclose(
            replayed.modelfit_best_fit, fitted.modelfit_best_fit, rtol=1e-5
        )
        assert code.count(".xlm.modelfit(") == 1
        axes = namespace["axs"].flat[0]
        expected = fitted.modelfit_results.item()
        np.testing.assert_allclose(
            axes.lines[1].get_ydata(),
            expected.eval(x=axes.lines[1].get_xdata()),
            rtol=1e-5,
        )
        reference_axes = Figure().subplots()
        erlab.plotting.plot_fit(fitted, ax=reference_axes, sigma=0.95)
        assert len(axes.collections) == 1
        np.testing.assert_allclose(
            axes.collections[0].get_paths()[0].vertices,
            reference_axes.collections[0].get_paths()[0].vertices,
            rtol=1e-5,
        )

        picker = _dialogs._FigureSourcePickerDialog(manager)
        qtbot.addWidget(picker)
        item = _source_picker_item(picker, fit_uid)
        assert item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable
        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
        assert fit_uid in picker.selected_targets()
        empty_uid = manager.create_empty_figure(show=False)
        empty = manager._child_node(empty_uid).tool_window
        assert manager._figure_workflows._add_imagetool_sources_to_figure(
            empty_uid, (fit_uid,)
        )
        assert empty.source_states()[0].kind == "fit"
        assert not empty.tool_status.operations

        drag_uid = manager.create_empty_figure(show=False)
        drag_composer = manager._child_node(drag_uid).tool_window
        model = manager.tree_view.model()
        mime = model.mimeData([model._row_index(fit_uid)])
        assert manager.tree_view.figure_source_uids_from_mime(mime) == (fit_uid,)
        assert drag_composer._source_drop_available(mime)
        assert drag_composer._add_sources_from_mime(mime)
        assert drag_composer.source_states()[0].kind == "fit"
        assert not drag_composer.tool_status.operations
        assert manager._figure_workflows._add_imagetool_sources_to_figure(
            drag_uid, (0,)
        )
        for source_kind in ("array", "fit"):
            replacement_dialog = _dialogs._AppendFigureTargetDialog(
                manager,
                (drag_uid,),
                None,
                allow_new_figure=True,
                source_kind=source_kind,
            )
            qtbot.addWidget(replacement_dialog)
            offered = tuple(
                replacement_dialog.source_combo.itemData(index)
                for index in range(replacement_dialog.source_combo.count())
            )
            expected = tuple(
                source.name
                for source in drag_composer.source_states()
                if source.kind == source_kind
            )
            assert offered == expected

        manager.tree_view.clearSelection()
        select_child_tool(manager, fit_uid)
        assert manager._selected_figure_source_targets() == [fit_uid]
        before = tuple(manager._figure_uids())
        monkeypatch.setattr(
            _dialogs._AppendFigureTargetDialog,
            "exec",
            lambda _dialog: QtWidgets.QDialog.DialogCode.Rejected,
        )
        manager.create_figure_from_selection()
        assert tuple(manager._figure_uids()) == before

        def accept_new_figure(dialog):
            dialog.action_combo.setCurrentIndex(
                dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_NEW)
            )
            return QtWidgets.QDialog.DialogCode.Accepted

        monkeypatch.setattr(
            _dialogs._AppendFigureTargetDialog, "exec", accept_new_figure
        )
        manager.create_figure_from_selection()
        new_uid = next(uid for uid in manager._figure_uids() if uid not in before)
        selected_composer = manager._child_node(new_uid).tool_window
        assert selected_composer.source_states()[0].kind == "fit"
        assert selected_composer.tool_status.operations[0].sources == (
            fit_tool.figure_fit_source_name(),
        )

        fit_tool._set_current_index(1)
        assert manager._figure_workflows._can_refresh_figure_source(
            figure_uid, source.name
        )
        assert manager._figure_workflows._refresh_figure_source(figure_uid, source.name)
        assert fit_tool._current_idx == 1
        stored = composer.source_payloads()[source.name]
        restored = erlab.interactive.utils.ToolWindow.from_dataset(
            composer.to_dataset()
        )
        qtbot.addWidget(restored)
        xr.testing.assert_identical(
            restored.source_payloads()[source.name].native_dataset,
            stored.native_dataset,
        )
        manager._remove_childtool(fit_uid)
        assert not manager._figure_workflows._can_refresh_figure_source(
            figure_uid, source.name
        )
        xr.testing.assert_identical(
            composer.source_payloads()[source.name].native_dataset,
            stored.native_dataset,
        )
