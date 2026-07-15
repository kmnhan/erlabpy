# ruff: noqa: F403, F405

import erlab.interactive._figurecomposer._ui._color_widgets as color_widgets
from erlab.interactive._figurecomposer._operations._method import (
    _catalog as method_catalog,
)
from erlab.interactive._figurecomposer._operations._method import (
    _editor as method_editor,
)
from erlab.interactive._figurecomposer._operations._method import (
    _plot_data as method_plot_data,
)
from erlab.interactive._figurecomposer._operations._method import (
    _plot_editor as method_plot_editor,
)
from erlab.interactive._figurecomposer._operations._method import _state as method_state

from ._common import *


def test_figure_composer_axes_plot_data_update_helper(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    args=((0.0, 1.0),),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    method_plot_editor._update_current_plot_data_arg(tool.operation_editor, "x", "0, 1")
    assert tool.tool_status.operations[0].method_args == ((0, 1), (0.0, 1.0))

    method_plot_editor._update_current_plot_data_arg(tool.operation_editor, "y", "2, 3")
    assert tool.tool_status.operations[0].method_args == ((0, 1), (2, 3))

    method_plot_editor._update_current_plot_data_arg(tool.operation_editor, "x", "")
    assert tool.tool_status.operations[0].method_args == ((2, 3),)


def test_figure_composer_axes_plot_method_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            setup=FigureSubplotsState(nrows=1, ncols=2),
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": (
                            (0.0, 0.5, 1.0),
                            (1.0, 0.5, 0.0),
                        ),
                        "method_kwargs": {
                            "color": "C1",
                            "linestyle": "--",
                            "linewidth": 2.5,
                            "marker": "o",
                            "markersize": 4.0,
                            "markerfacecolor": "white",
                            "markeredgecolor": "black",
                            "alpha": 0.75,
                            "label": "manual",
                            "zorder": 4.0,
                            "clip_on": False,
                            "transform": "ignored",
                        },
                        "method_transform": "blend",
                        "method_transform_x": "data",
                        "method_transform_y": "axes",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 1),)),
                ).model_copy(
                    update={
                        "method_args": ((0.0, 0.25, 1.0),),
                        "method_transform": "custom",
                        "method_transform_expression": "ax.transAxes",
                        "trusted": True,
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
    method_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCombo"
    )
    x_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit"
    )
    y_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotYEdit"
    )
    color_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotColorEdit"
    )
    color_button = method_page.findChild(
        QtWidgets.QWidget, "figureComposerAxesMethodPlotColorEditButton"
    )
    style_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotLineStyleCombo"
    )
    width_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodPlotLineWidthSpin"
    )
    marker_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotMarkerCombo"
    )
    transform_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    transform_x_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformXCombo"
    )
    transform_y_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformYCombo"
    )
    kwargs_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit"
    )
    assert method_combo is not None
    assert x_edit is not None
    assert y_edit is not None
    assert color_edit is not None
    assert color_button is not None
    assert style_combo is not None
    assert width_spin is not None
    assert marker_combo is not None
    assert transform_combo is not None
    assert transform_x_combo is not None
    assert transform_y_combo is not None
    assert kwargs_edit is not None
    assert method_combo.currentData() == "plot"
    assert x_edit.text() == "0.0, 0.5, 1.0"
    assert y_edit.text() == "1.0, 0.5, 0.0"
    assert color_edit.text() == "C1"
    assert width_spin.value() == pytest.approx(2.5)
    assert kwargs_edit.text() == "clip_on=False"

    color_edit.setText("tab:blue")
    color_edit.setModified(True)
    color_widget = color_edit.parentWidget()
    assert isinstance(color_widget, color_widgets._ColorLineEditWidget)
    color_widget.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "tab:blue"
    kwargs_edit.setText('clip_on=True, transform="ax.transData"')
    kwargs_edit.setModified(True)
    kwargs_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["clip_on"] is True
    assert tool.tool_status.operations[0].method_kwargs["transform"] == "ignored"

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(1)
    )
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
    custom_transform_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerMethodTransformExpressionEdit"
    )
    custom_transform_trusted = method_page.findChild(
        QtWidgets.QCheckBox, "figureComposerMethodTransformTrustedCheck"
    )
    assert custom_transform_edit is not None
    assert custom_transform_trusted is not None
    assert custom_transform_edit.text() == "ax.transAxes"
    assert custom_transform_trusted.isChecked()

    fig = tool.figure
    figurecomposer_rendering._render_into_figure(tool, fig, sync_visible=False)
    assert tool._operation_render_errors == {}
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), (0.0, 0.5, 1.0))
    np.testing.assert_allclose(line.get_ydata(), (1.0, 0.5, 0.0))
    assert line.get_color() == "tab:blue"
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() == pytest.approx(2.5)
    assert line.get_marker() == "o"
    assert line.get_markersize() == pytest.approx(4.0)
    assert line.get_markerfacecolor() == "white"
    assert line.get_markeredgecolor() == "black"
    assert line.get_alpha() == pytest.approx(0.75)
    assert line.get_label() == "manual"
    assert line.get_zorder() == pytest.approx(4.0)
    assert line.get_clip_on() is True
    assert isinstance(line.get_transform(), mtransforms.BlendedGenericTransform)
    assert fig.axes[1].lines[0].get_transform() == fig.axes[1].transAxes

    code = tool.generated_code()
    assert "import matplotlib.transforms as mtransforms" in code
    assert "axs[0, 0].plot((0.0, 0.5, 1.0), (1.0, 0.5, 0.0)" in code
    assert 'color="tab:blue"' in code
    assert "clip_on=True" in code
    assert "transform=ax.transData" not in code
    assert (
        "transform=mtransforms.blended_transform_factory("
        "axs[0, 0].transData, axs[0, 0].transAxes)"
    ) in code
    assert "ax.plot((0.0, 0.25, 1.0), transform=ax.transAxes)" in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    axs = namespace["axs"]
    assert len(axs[0, 0].lines) == 1
    assert axs[0, 0].lines[0].get_color() == "tab:blue"
    assert axs[0, 1].lines[0].get_transform() == axs[0, 1].transAxes


def test_figure_composer_axes_errorbar_method_render_and_codegen(qtbot) -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": (
                            (0.0, 0.5, 1.0),
                            (1.0, 0.5, 0.0),
                        ),
                        "method_kwargs": {
                            "xerr": (0.05, 0.1, 0.15),
                            "yerr": 0.2,
                            "color": "C1",
                            "linestyle": "--",
                            "marker": "o",
                            "capsize": 3.0,
                            "label": "manual errorbar",
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
    method_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodCombo"
    )
    x_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarXEdit"
    )
    y_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarYEdit"
    )
    xerr_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarXErrorEdit"
    )
    yerr_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodErrorbarYErrorEdit"
    )
    capsize_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodErrorbarCapSizeSpin"
    )
    kwargs_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodKwEdit"
    )
    assert method_combo is not None
    assert x_edit is not None
    assert y_edit is not None
    assert xerr_edit is not None
    assert yerr_edit is not None
    assert capsize_spin is not None
    assert kwargs_edit is not None
    assert method_combo.currentData() == "errorbar"
    assert x_edit.text() == "0.0, 0.5, 1.0"
    assert y_edit.text() == "1.0, 0.5, 0.0"
    assert xerr_edit.text() == "0.05, 0.1, 0.15"
    assert yerr_edit.text() == "0.2"
    assert capsize_spin.value() == pytest.approx(3.0)
    assert kwargs_edit.text() == ""

    xerr_edit.setText("0.1, 0.2, 0.3")
    xerr_edit.setModified(True)
    xerr_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["xerr"] == (0.1, 0.2, 0.3)
    capsize_spin.setValue(4.0)
    assert tool.tool_status.operations[0].method_kwargs["capsize"] == pytest.approx(4.0)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    container = tool.figure.axes[0].containers[0]
    assert isinstance(container, ErrorbarContainer)
    line = container.lines[0]
    np.testing.assert_allclose(
        np.asarray(line.get_xdata(), dtype=float), (0.0, 0.5, 1.0)
    )
    np.testing.assert_allclose(
        np.asarray(line.get_ydata(), dtype=float), (1.0, 0.5, 0.0)
    )
    assert line.get_color() == "C1"
    assert line.get_linestyle() == "--"
    assert line.get_marker() == "o"
    assert container.get_label() == "manual errorbar"
    _assert_errorbar_xerr(container, (0.0, 0.5, 1.0), (1.0, 0.5, 0.0), (0.1, 0.2, 0.3))
    _assert_errorbar_capsize(container, 4.0)

    code = tool.generated_code()
    assert "axs[0, 0].errorbar((0.0, 0.5, 1.0), (1.0, 0.5, 0.0)" in code
    assert "xerr=(0.1, 0.2, 0.3)" in code
    assert "yerr=0.2" in code
    assert "capsize=4.0" in code

    namespace: dict[str, typing.Any] = {}
    exec(code, namespace)  # noqa: S102
    generated_container = namespace["axs"][0, 0].containers[0]
    assert isinstance(generated_container, ErrorbarContainer)
    generated_line = generated_container.lines[0]
    np.testing.assert_allclose(
        np.asarray(generated_line.get_xdata(), dtype=float), (0.0, 0.5, 1.0)
    )
    np.testing.assert_allclose(
        np.asarray(generated_line.get_ydata(), dtype=float), (1.0, 0.5, 0.0)
    )
    _assert_errorbar_xerr(
        generated_container,
        (0.0, 0.5, 1.0),
        (1.0, 0.5, 0.0),
        (0.1, 0.2, 0.3),
    )
    _assert_errorbar_capsize(generated_container, 4.0)


@pytest.mark.parametrize(
    "method_args",
    [
        ((1.0, 0.5, 0.0),),
        ((), (1.0, 0.5, 0.0)),
        ((0.0, 0.5, 1.0), ()),
    ],
)
def test_figure_composer_axes_errorbar_method_requires_entered_x_and_y(
    qtbot,
    method_args: tuple[tuple[float, ...], ...],
) -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_args": method_args,
                        "method_kwargs": {
                            "xerr": (0.1, 0.2, 0.3),
                            "yerr": (0.2, 0.2, 0.2),
                            "marker": "o",
                        },
                    }
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert "Enter X and Y values for ax.errorbar" in errors[0]
    with pytest.raises(ValueError, match=r"Enter X and Y values for ax\.errorbar"):
        tool.generated_code()


def test_figure_composer_axes_plot_data_mode_switches_editor(qtbot) -> None:
    data = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
            ),
            primary_source="data",
        ),
        source_data={"data": data},
    )
    qtbot.addWidget(tool)

    tool.operation_panel.operation_list.setCurrentItem(
        tool.operation_panel.operation_list.topLevelItem(0)
    )
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
    mode_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotDataModeCombo"
    )
    assert mode_combo is not None
    assert mode_combo.currentData() == "entered"
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit")
        is not None
    )
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotYEdit")
        is not None
    )

    _activate_combo_index(mode_combo, mode_combo.findData("from_data"))
    qtbot.wait_until(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QComboBox, "figureComposerAxesMethodPlotYSourceCombo"
            )
            is not None
        ),
        timeout=5000,
    )

    operation = tool.tool_status.operations[0]
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    method_page = tool.operation_editor.stack.currentWidget()
    assert (
        method_page.findChild(QtWidgets.QLineEdit, "figureComposerAxesMethodPlotXEdit")
        is None
    )
    x_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotXSourceCombo"
    )
    x_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotXValuesCombo"
    )
    y_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotYSourceCombo"
    )
    y_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodPlotYValuesCombo"
    )
    assert x_source_combo is not None
    assert x_values_combo is not None
    assert y_source_combo is not None
    assert y_values_combo is not None
    assert x_source_combo.property("figureComposerPlotDataRole") == "x_source"
    assert x_values_combo.property("figureComposerPlotDataRole") == "x_values"
    assert y_source_combo.property("figureComposerPlotDataRole") == "y_source"
    assert y_values_combo.property("figureComposerPlotDataRole") == "y_values"
    assert x_source_combo.toolTip()
    assert x_values_combo.toolTip()
    assert y_source_combo.toolTip()
    assert y_values_combo.toolTip()
    assert x_source_combo.toolTip() != x_values_combo.toolTip()
    assert y_source_combo.toolTip() != y_values_combo.toolTip()
    assert x_source_combo.itemText(0) != x_values_combo.itemText(0)
    assert x_source_combo.currentData() is None
    assert y_source_combo.currentData() == "data"
    y_value_options = {
        y_values_combo.itemData(index) for index in range(y_values_combo.count())
    }
    assert y_value_options >= {
        ("data", None),
        ("coord", "kx"),
    }
    assert all(
        y_values_combo.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole)
        for index in range(y_values_combo.count())
        if y_values_combo.itemData(index) is not None
    )

    method_editor._update_current_method_name(tool.operation_editor, "errorbar")
    tool.operation_editor.select_section("method")
    qtbot.wait_until(
        lambda: (
            tool.operation_editor.stack.currentWidget().findChild(
                QtWidgets.QComboBox,
                "figureComposerAxesMethodErrorbarXErrorSourceCombo",
            )
            is not None
        ),
        timeout=5000,
    )
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "errorbar"
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    method_page = tool.operation_editor.stack.currentWidget()
    xerr_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarXErrorSourceCombo"
    )
    xerr_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarXErrorValuesCombo"
    )
    yerr_source_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarYErrorSourceCombo"
    )
    yerr_values_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerAxesMethodErrorbarYErrorValuesCombo"
    )
    assert xerr_source_combo is not None
    assert xerr_values_combo is not None
    assert yerr_source_combo is not None
    assert yerr_values_combo is not None
    assert xerr_source_combo.property("figureComposerPlotDataRole") == "xerr_source"
    assert xerr_values_combo.property("figureComposerPlotDataRole") == "xerr_values"
    assert yerr_source_combo.property("figureComposerPlotDataRole") == "yerr_source"
    assert yerr_values_combo.property("figureComposerPlotDataRole") == "yerr_values"
    assert xerr_source_combo.currentData() is None
    assert yerr_source_combo.currentData() is None

    method_editor._update_current_method_name(tool.operation_editor, "plot")
    operation = tool.tool_status.operations[0]
    assert operation.method_name == "plot"
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_xerr is None
    assert operation.method_plot_yerr is None


def test_figure_composer_axes_plot_method_picks_data_render_and_codegen(qtbot) -> None:
    x_source = xr.DataArray(
        np.array([10.0, 20.0, 30.0]),
        dims=("kx",),
        coords={"kx": [0.0, 0.5, 1.0]},
        name="x_source",
    )
    y_source = xr.DataArray(
        np.array([1.0, 4.0, 9.0]),
        dims=("point",),
        coords={"point": [0, 1, 2]},
        name="y_source",
    )
    tool = FigureComposerTool(
        y_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="x_data", label="x data"),
                FigureSourceState(name="y_data", label="y data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": FigureMethodPlotValueState(
                            source="x_data", kind="coord", name="kx"
                        ),
                        "method_plot_y": FigureMethodPlotValueState(
                            source="y_data", kind="data"
                        ),
                        "method_kwargs": {"color": "C2", "label": "picked"},
                    }
                ),
            ),
            primary_source="y_data",
        ),
        source_data={"x_data": x_source, "y_data": y_source},
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    line = tool.figure.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), x_source.coords["kx"].values)
    np.testing.assert_allclose(line.get_ydata(), y_source.values)
    assert line.get_color() == "C2"
    assert line.get_label() == "picked"

    code = tool.generated_code()
    assert 'x_data.coords["kx"].values' in code
    assert "y_data.values" in code
    namespace: dict[str, typing.Any] = {
        "x_data": x_source,
        "y_data": y_source,
    }
    exec(code, namespace)  # noqa: S102
    generated_line = namespace["axs"][0, 0].lines[0]
    np.testing.assert_allclose(generated_line.get_xdata(), x_source.coords["kx"].values)
    np.testing.assert_allclose(generated_line.get_ydata(), y_source.values)


def test_figure_composer_axes_errorbar_method_picks_data_render_and_codegen(
    qtbot,
) -> None:
    values_source = xr.DataArray(
        np.array([0.1, 0.2, 0.3]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="values",
    )
    stderr_source = xr.DataArray(
        np.array([0.01, 0.02, 0.03]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="stderr",
    )
    yerr_source = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="yerr",
    )
    tool = FigureComposerTool(
        values_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="values_data", label="values data"),
                FigureSourceState(name="stderr_data", label="stderr data"),
                FigureSourceState(name="yerr_data", label="y error data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": FigureMethodPlotValueState(
                            source="values_data", kind="data"
                        ),
                        "method_plot_y": FigureMethodPlotValueState(
                            source="values_data", kind="coord", name="scan"
                        ),
                        "method_plot_xerr": FigureMethodPlotValueState(
                            source="stderr_data", kind="data"
                        ),
                        "method_plot_yerr": FigureMethodPlotValueState(
                            source="yerr_data", kind="data"
                        ),
                        "method_kwargs": {
                            "xerr": 99.0,
                            "yerr": 99.0,
                            "color": "C3",
                            "label": "picked errorbar",
                            "linestyle": "none",
                            "marker": "o",
                        },
                    }
                ),
            ),
            primary_source="values_data",
        ),
        source_data={
            "values_data": values_source,
            "stderr_data": stderr_source,
            "yerr_data": yerr_source,
        },
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    assert tool._operation_render_errors == {}
    container = tool.figure.axes[0].containers[0]
    assert isinstance(container, ErrorbarContainer)
    line = container.lines[0]
    np.testing.assert_allclose(line.get_xdata(), values_source.values)
    np.testing.assert_allclose(line.get_ydata(), values_source.coords["scan"].values)
    assert line.get_color() == "C3"
    assert container.get_label() == "picked errorbar"
    _assert_errorbar_xerr(
        container,
        values_source.values,
        values_source.coords["scan"].values,
        stderr_source.values,
    )
    _assert_errorbar_yerr(
        container,
        values_source.values,
        values_source.coords["scan"].values,
        yerr_source.values,
    )

    code = tool.generated_code()
    assert "values_data.values" in code
    assert 'values_data.coords["scan"].values' in code
    assert "xerr=stderr_data.values" in code
    assert "yerr=yerr_data.values" in code
    assert "xerr=99.0" not in code
    assert "yerr=99.0" not in code
    namespace: dict[str, typing.Any] = {
        "values_data": values_source,
        "stderr_data": stderr_source,
        "yerr_data": yerr_source,
    }
    exec(code, namespace)  # noqa: S102
    generated_container = namespace["axs"][0, 0].containers[0]
    assert isinstance(generated_container, ErrorbarContainer)
    generated_line = generated_container.lines[0]
    np.testing.assert_allclose(generated_line.get_xdata(), values_source.values)
    np.testing.assert_allclose(
        generated_line.get_ydata(), values_source.coords["scan"].values
    )
    _assert_errorbar_xerr(
        generated_container,
        values_source.values,
        values_source.coords["scan"].values,
        stderr_source.values,
    )
    _assert_errorbar_yerr(
        generated_container,
        values_source.values,
        values_source.coords["scan"].values,
        yerr_source.values,
    )


def test_figure_composer_axes_errorbar_method_requires_picked_x_and_y(
    qtbot,
) -> None:
    values_source = xr.DataArray(
        np.array([1.0, 4.0, 9.0]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="values",
    )
    stderr_source = xr.DataArray(
        np.array([0.1, 0.2, 0.3]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="stderr",
    )
    yerr_source = xr.DataArray(
        np.array([0.4, 0.5, 0.6]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0, 30.0]},
        name="yerr",
    )
    tool = FigureComposerTool(
        values_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="values_data", label="values data"),
                FigureSourceState(name="stderr_data", label="stderr data"),
                FigureSourceState(name="yerr_data", label="y error data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                    axes=FigureAxesSelectionState(axes=((0, 0),)),
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_y": FigureMethodPlotValueState(
                            source="values_data", kind="data"
                        ),
                        "method_plot_xerr": FigureMethodPlotValueState(
                            source="stderr_data", kind="data"
                        ),
                        "method_plot_yerr": FigureMethodPlotValueState(
                            source="yerr_data", kind="data"
                        ),
                    }
                ),
            ),
            primary_source="values_data",
        ),
        source_data={
            "values_data": values_source,
            "stderr_data": stderr_source,
            "yerr_data": yerr_source,
        },
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert "Choose X values for ax.errorbar" in errors[0]
    with pytest.raises(ValueError, match=r"Choose X values for ax\.errorbar"):
        tool.generated_code()


@pytest.mark.parametrize(
    ("source_data", "x_state", "y_state", "message"),
    [
        (
            {"data": xr.DataArray([1.0, 2.0], dims=("x",), name="data")},
            None,
            FigureMethodPlotValueState(source="data", kind="coord", name="missing"),
            "Coordinate 'missing' is not available",
        ),
        (
            {"image": xr.DataArray(np.ones((2, 2)), dims=("x", "y"), name="image")},
            None,
            FigureMethodPlotValueState(source="image", kind="data"),
            "one-dimensional",
        ),
        (
            {
                "x": xr.DataArray([0.0, 1.0, 2.0], dims=("x",), name="x"),
                "y": xr.DataArray([1.0, 2.0], dims=("y",), name="y"),
            },
            FigureMethodPlotValueState(source="x", kind="data"),
            FigureMethodPlotValueState(source="y", kind="data"),
            "same length",
        ),
        (
            {"data": xr.DataArray([1.0, 2.0], dims=("x",), name="data")},
            None,
            FigureMethodPlotValueState(source="missing", kind="data"),
            "is not available",
        ),
    ],
)
def test_figure_composer_axes_plot_picked_data_invalid_inputs(
    qtbot,
    source_data: dict[str, xr.DataArray],
    x_state: FigureMethodPlotValueState | None,
    y_state: FigureMethodPlotValueState,
    message: str,
) -> None:
    primary_source = next(iter(source_data))
    tool = FigureComposerTool(
        source_data[primary_source],
        recipe=FigureRecipeState(
            sources=tuple(
                FigureSourceState(name=name, label=name) for name in source_data
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": x_state,
                        "method_plot_y": y_state,
                    }
                ),
            ),
            primary_source=primary_source,
        ),
        source_data=source_data,
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert message in errors[0]
    with pytest.raises(ValueError, match=message):
        tool.generated_code()


@pytest.mark.parametrize("axis", ["xerr", "yerr"])
def test_figure_composer_axes_errorbar_picked_error_data_invalid_lengths(
    qtbot,
    axis: str,
) -> None:
    values_source = xr.DataArray(
        np.array([0.1, 0.2]),
        dims=("scan",),
        coords={"scan": [10.0, 20.0]},
        name="values",
    )
    stderr_source = xr.DataArray(
        np.array([0.01, 0.02, 0.03]),
        dims=("scan",),
        name="stderr",
    )
    tool = FigureComposerTool(
        values_source,
        recipe=FigureRecipeState(
            sources=(
                FigureSourceState(name="values_data", label="values data"),
                FigureSourceState(name="stderr_data", label="stderr data"),
            ),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="errorbar",
                ).model_copy(
                    update={
                        "method_plot_data_mode": "from_data",
                        "method_plot_x": FigureMethodPlotValueState(
                            source="values_data", kind="data"
                        ),
                        "method_plot_y": FigureMethodPlotValueState(
                            source="values_data", kind="coord", name="scan"
                        ),
                        f"method_plot_{axis}": FigureMethodPlotValueState(
                            source="stderr_data", kind="data"
                        ),
                    }
                ),
            ),
            primary_source="values_data",
        ),
        source_data={"values_data": values_source, "stderr_data": stderr_source},
    )
    qtbot.addWidget(tool)

    figurecomposer_rendering._render_into_figure(tool, tool.figure, sync_visible=False)
    errors = tuple(tool._operation_render_errors.values())
    assert len(errors) == 1
    assert "same length" in errors[0]
    with pytest.raises(ValueError, match="same length"):
        tool.generated_code()


def test_figure_composer_axes_plot_picked_data_sources_are_renamed() -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_x": FigureMethodPlotValueState(
                source="x_data", kind="coord", name="kx"
            ),
            "method_plot_y": FigureMethodPlotValueState(source="y_data", kind="data"),
        }
    )

    assert figurecomposer_operation_metadata.declared_operation_source_names(
        operation
    ) == ("x_data", "y_data")
    renamed = figurecomposer_operation_metadata.rename_operation_sources(
        operation, {"x_data": "x_copy", "y_data": "y_copy"}
    )

    assert renamed.method_plot_x == FigureMethodPlotValueState(
        source="x_copy", kind="coord", name="kx"
    )
    assert renamed.method_plot_y == FigureMethodPlotValueState(
        source="y_copy", kind="data"
    )

    errorbar_operation = operation.model_copy(
        update={
            "method_name": "errorbar",
            "method_plot_xerr": FigureMethodPlotValueState(
                source="stderr_data", kind="data"
            ),
            "method_plot_yerr": FigureMethodPlotValueState(
                source="stderr_data", kind="coord", name="kx"
            ),
        }
    )
    assert figurecomposer_operation_metadata.declared_operation_source_names(
        errorbar_operation
    ) == (
        "x_data",
        "y_data",
        "stderr_data",
    )
    renamed_errorbar = figurecomposer_operation_metadata.rename_operation_sources(
        errorbar_operation,
        {"x_data": "x_copy", "y_data": "y_copy", "stderr_data": "stderr_copy"},
    )
    assert renamed_errorbar.method_plot_xerr == FigureMethodPlotValueState(
        source="stderr_copy", kind="data"
    )
    assert renamed_errorbar.method_plot_yerr == FigureMethodPlotValueState(
        source="stderr_copy", kind="coord", name="kx"
    )


def test_figure_composer_axes_plot_data_helper_edges() -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(1, 6),
        dims=("scan", "point"),
        coords={"scan": [0], "point": np.linspace(0.0, 1.0, 6)},
        name="source",
    )
    coord_grid = xr.DataArray(
        np.ones((1, 6)),
        dims=("scan", "point"),
        coords=source.coords,
        name="coord_grid",
    )
    source_with_grid_coord = source.assign_coords(grid=coord_grid)
    source_with_2d_coord = xr.DataArray(
        np.ones((2, 6)),
        dims=("scan", "point"),
        coords={"scan": [0, 1], "point": np.linspace(0.0, 1.0, 6)},
        name="source_with_2d_coord",
    ).assign_coords(
        grid=(
            ("scan", "point"),
            np.arange(12.0).reshape(2, 6),
        )
    )
    source_data = {"source": source_with_grid_coord}

    assert method_plot_editor._plot_data_mode_text("unknown") == "Enter values"
    assert method_plot_editor._plot_value_display(None) == "Choose values"
    assert method_plot_data._plot_value_options(source_data, None) == ()
    assert method_plot_data._plot_value_options(source_data, "missing") == ()
    option_source_data = {"source": source_with_2d_coord}
    assert ("coord", "grid") not in {
        value
        for _text, value in method_plot_data._plot_value_options(
            option_source_data, "source"
        )
    }
    assert method_plot_data._plot_coord_by_name(source, "point") is not None
    assert method_plot_editor._plot_source_combo_tooltip("x", has_sources=False)
    xerr_source_tooltip = method_plot_editor._plot_source_combo_tooltip(
        "xerr", has_sources=True
    )
    yerr_source_tooltip = method_plot_editor._plot_source_combo_tooltip(
        "yerr", has_sources=True
    )
    assert xerr_source_tooltip.replace("x error", "error") == (
        yerr_source_tooltip.replace("y error", "error")
    )
    assert method_plot_editor._plot_values_combo_tooltip(
        "y", source="source", value_options_match=False
    )
    xerr_values_tooltip = method_plot_editor._plot_values_combo_tooltip(
        "xerr", source="source", value_options_match=True
    )
    yerr_values_tooltip = method_plot_editor._plot_values_combo_tooltip(
        "yerr", source="source", value_options_match=True
    )
    assert xerr_values_tooltip.replace("x error", "error") == (
        yerr_values_tooltip.replace("y error", "error")
    )
    with pytest.raises(ValueError, match="Unknown plot value selection"):
        method_plot_editor._plot_value_combo_data_parts(("bad", None))

    data_state = FigureMethodPlotValueState(source="source", kind="data")
    code, value = method_plot_data._plot_value_code_and_data(source_data, data_state)
    assert str(code) == "source.squeeze(drop=True).values"
    assert value.dims == ("point",)

    coord_state = FigureMethodPlotValueState(
        source="source", kind="coord", name="point"
    )
    coord_code, coord_value = method_plot_data._plot_value_code_and_data(
        source_data, coord_state
    )
    assert str(coord_code) == 'source.coords["point"].values'
    assert coord_value.dims == ("point",)

    with pytest.raises(ValueError, match="Choose a coordinate"):
        method_plot_data._plot_value_data(
            source_data,
            FigureMethodPlotValueState(source="source", kind="coord"),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        method_plot_data._plot_value_data(
            option_source_data,
            FigureMethodPlotValueState(source="source", kind="coord", name="grid"),
        )
    with pytest.raises(ValueError, match="Choose Y values"):
        method_plot_data._picked_plot_args(
            source_data,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            ).model_copy(update={"method_plot_data_mode": "from_data"}),
            method_catalog.AXES_METHODS["plot"],
        )

    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "from_data",
            "method_plot_y": data_state,
        }
    )
    args = method_plot_data._picked_plot_args(
        source_data, operation, method_catalog.AXES_METHODS["plot"]
    )
    assert len(args) == 1
    np.testing.assert_allclose(args[0], source.values.reshape(-1))
    code_args = method_plot_data._picked_plot_code_args(
        source_data, operation, method_catalog.AXES_METHODS["plot"]
    )
    assert tuple(str(arg) for arg in code_args) == ("source.squeeze(drop=True).values",)


def test_figure_composer_axes_plot_data_update_helpers(qtbot) -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "method_plot_data_mode": "entered",
            "method_plot_x": FigureMethodPlotValueState(source="old", kind="data"),
        }
    )
    data = xr.DataArray([1.0, 2.0], dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)
    method_plot_editor._update_current_plot_data_mode(tool.operation_editor, "invalid")
    assert tool.tool_status.operations[0] == operation

    method_plot_editor._update_current_plot_data_mode(
        tool.operation_editor, "from_data"
    )
    assert tool.tool_status.operations[0].method_plot_data_mode == "from_data"
    assert tool.tool_status.operations[0].method_plot_y == FigureMethodPlotValueState(
        source="data", kind="data"
    )

    method_plot_editor._update_current_plot_value_source(
        tool.operation_editor, "x", None
    )
    assert tool.tool_status.operations[0].method_plot_x is None
    method_plot_editor._update_current_plot_value_source(
        tool.operation_editor, "x", "data"
    )
    assert tool.tool_status.operations[0].method_plot_x == FigureMethodPlotValueState(
        source="data", kind="data"
    )
    method_plot_editor._update_current_plot_value_selection(
        tool.operation_editor, "x", ("coord", "x")
    )
    assert tool.tool_status.operations[0].method_plot_x == FigureMethodPlotValueState(
        source="data", kind="coord", name="x"
    )
    method_plot_editor._update_current_plot_value_selection(
        tool.operation_editor, "x", None
    )
    assert tool.tool_status.operations[0].method_plot_x is None

    assert (
        figurecomposer_operation_metadata.declared_operation_source_names(
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="missing",
            )
        )
        == ()
    )


def test_figure_composer_axes_plot_optional_bool_editor_branch(qtbot) -> None:
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="tick_params",
        kwargs={"reset": True},
    )
    layout_parent = QtWidgets.QWidget()
    qtbot.addWidget(layout_parent)
    layout = QtWidgets.QFormLayout(layout_parent)
    data = xr.DataArray([1.0, 2.0], dims=("x",), name="data")
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(operation,),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    control = method_catalog._optional_bool_kwarg_combo(
        "Reset",
        "reset",
        "figureComposerTestResetCombo",
        "tooltip",
    )
    method_editor._add_method_control_row(
        tool.operation_editor,
        layout,
        operation,
        method_catalog.AXES_METHODS["tick_params"],
        control,
    )
    combo = layout_parent.findChild(QtWidgets.QComboBox, "figureComposerTestResetCombo")
    assert combo is not None
    assert combo.currentData() == "True"

    combo.setCurrentText("False")
    combo.activated.emit(combo.currentIndex())
    assert tool.tool_status.operations[0].method_kwargs["reset"] is False


def test_figure_composer_axes_plot_combo_helper_edges(qtbot) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(1, 6),
        dims=("scan", "point"),
        coords={
            "scan": [0],
            "point": np.linspace(0.0, 1.0, 6),
            "scan_coord": ("scan", [10.0]),
            "row_coord": (("scan", "point"), np.arange(6.0).reshape(1, 6)),
        },
        name="source",
    )

    tool = FigureComposerTool(
        source,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="source", label="display source"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ),
            ),
            primary_source="source",
        ),
    )
    qtbot.addWidget(tool)
    changed: list[object] = []

    mode_combo = method_plot_editor._plot_data_mode_combo(
        tool.operation_editor,
        current=None,
        changed=changed.append,
        parent=tool.operation_editor,
        mixed=True,
    )
    assert mode_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", mode_combo.model()).item(0).isEnabled()

    source_combo = method_plot_editor._plot_source_combo(
        tool.operation_editor,
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
        mixed=True,
    )
    assert source_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", source_combo.model()).item(0).isEnabled()

    values_combo = method_plot_editor._plot_values_combo(
        tool.operation_editor,
        source="source",
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
        mixed=True,
    )
    assert values_combo.currentData() is _editor_controls.MIXED_VALUE
    assert not typing.cast("typing.Any", values_combo.model()).item(0).isEnabled()

    no_source_values = method_plot_editor._plot_values_combo(
        tool.operation_editor,
        source=None,
        current=None,
        changed=changed.append,
        axis="y",
        parent=tool.operation_editor,
        allow_none=False,
    )
    assert no_source_values.itemData(0) is None
    assert not no_source_values.isEnabled()
    assert method_plot_editor._plot_values_combo_tooltip(
        "y", source=None, value_options_match=True
    )

    squeezed_coord_state = FigureMethodPlotValueState(
        source="source", kind="coord", name="row_coord"
    )
    code, value = method_plot_data._plot_value_code_and_data(
        tool._document.source_data, squeezed_coord_state
    )
    assert str(code) == 'source.coords["row_coord"].squeeze(drop=True).values'
    assert value.dims == ("point",)

    with pytest.raises(ValueError, match="Choose a coordinate"):
        method_plot_data._plot_value_code_and_data(
            tool._document.source_data,
            FigureMethodPlotValueState(source="source", kind="coord"),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        method_plot_data._plot_value_code_and_data(
            tool._document.source_data,
            FigureMethodPlotValueState(
                source="source", kind="coord", name="scan_coord"
            ),
        )
    with pytest.raises(ValueError, match="Choose Y values"):
        method_plot_data._picked_plot_code_args(
            tool._document.source_data,
            FigureOperationState.method(
                family=FigureMethodFamily.AXES,
                name="plot",
            ).model_copy(update={"method_plot_data_mode": "from_data"}),
            method_catalog.AXES_METHODS["plot"],
        )

    callback = method_editor._method_optional_bool_kwarg_callback(
        tool.operation_editor, "visible"
    )
    callback(None)
    assert "visible" not in tool.tool_status.operations[0].method_kwargs
    callback("True")
    assert tool.tool_status.operations[0].method_kwargs["visible"] is True

    optional_bool_control = method_catalog.MethodControlSpec(
        kind=method_catalog.MethodControlKind.OPTIONAL_BOOL_KWARG_COMBO,
        label="Visible",
        tooltip="",
        object_name="visible",
    )
    assert method_state._control_accepts_value(optional_bool_control, None)
    assert method_state._control_accepts_value(optional_bool_control, False)
    assert not method_state._control_accepts_value(optional_bool_control, "False")


def test_figure_composer_batch_same_plot_method_edits_selected_steps(qtbot) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [0.0, 1.0], "ky": [0.0, 1.0]},
        name="data",
    )
    tool = FigureComposerTool(
        data,
        recipe=FigureRecipeState(
            sources=(FigureSourceState(name="data", label="data"),),
            operations=(
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_kwargs": {"color": "red", "linewidth": 1.0},
                        "method_transform": "axes",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="plot",
                ).model_copy(
                    update={
                        "method_kwargs": {"color": "blue", "linewidth": 3.0},
                        "method_transform": "figure",
                    }
                ),
                FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name="set_title",
                    args=("unchanged",),
                ),
            ),
            primary_source="data",
        ),
    )
    qtbot.addWidget(tool)

    _select_operation_rows(tool, (0, 1))
    tool.operation_editor.select_section("method")
    method_page = tool.operation_editor.stack.currentWidget()
    color_edit = method_page.findChild(
        QtWidgets.QLineEdit, "figureComposerAxesMethodPlotColorEdit"
    )
    transform_combo = method_page.findChild(
        QtWidgets.QComboBox, "figureComposerMethodTransformModeCombo"
    )
    width_spin = method_page.findChild(
        QtWidgets.QDoubleSpinBox, "figureComposerAxesMethodPlotLineWidthSpin"
    )
    assert color_edit is not None
    assert transform_combo is not None
    assert width_spin is not None
    assert color_edit.text() == ""
    assert color_edit.placeholderText() == "(multiple values)"
    assert transform_combo.currentIndex() == 0
    mixed_item = typing.cast("typing.Any", transform_combo.model()).item(0)
    assert mixed_item is not None
    assert not mixed_item.isEnabled()
    assert width_spin.value() == pytest.approx(float(mpl.rcParams["lines.linewidth"]))
    width_spin_container = width_spin.parentWidget()
    assert width_spin_container is not None
    assert width_spin_container.findChild(
        QtWidgets.QLabel, "figureComposerMixedValueMarker"
    )

    color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "red"
    assert tool.tool_status.operations[1].method_kwargs["color"] == "blue"
    assert tool.tool_status.operations[0].method_kwargs["linewidth"] == pytest.approx(
        1.0
    )
    assert tool.tool_status.operations[1].method_kwargs["linewidth"] == pytest.approx(
        3.0
    )

    color_edit.setText("tab:green")
    color_edit.setModified(True)
    color_edit.editingFinished.emit()
    assert tool.tool_status.operations[0].method_kwargs["color"] == "tab:green"
    assert tool.tool_status.operations[1].method_kwargs["color"] == "tab:green"
    width_spin.setValue(4.5)
    assert tool.tool_status.operations[0].method_kwargs["linewidth"] == pytest.approx(
        4.5
    )
    assert tool.tool_status.operations[1].method_kwargs["linewidth"] == pytest.approx(
        4.5
    )

    _activate_combo_text(transform_combo, "blend")
    assert tool.tool_status.operations[0].method_transform == "blend"
    assert tool.tool_status.operations[1].method_transform == "blend"
    assert tool.tool_status.operations[2].method_args == ("unchanged",)
    assert _selected_operation_rows(tool) == (0, 1)
