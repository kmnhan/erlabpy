import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure
from qtpy import QtWidgets

import erlab.interactive._figurecomposer._defaults as figurecomposer_defaults
import erlab.interactive._figurecomposer._tool as figurecomposer_tool_module
import erlab.interactive.utils
from erlab.interactive._figurecomposer import FigureComposerTool, FigureExportState
from erlab.interactive._figurecomposer._defaults import figure_options_context
from erlab.interactive._figurecomposer._ui._export_panel import FigureExportPanel
from erlab.interactive._options.core import model_with_workspace_overrides
from erlab.interactive._options.schema import AppOptions


def _data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )


def test_figure_export_state_inherits_and_migrates_legacy_values() -> None:
    assert FigureExportState() == FigureExportState(
        dpi="inherit",
        transparent="inherit",
        bbox_inches="inherit",
        pad_inches="inherit",
    )
    assert FigureExportState.model_validate(
        {
            "dpi": 300.0,
            "transparent": False,
            "bbox_inches": None,
        }
    ) == FigureExportState(
        dpi=300.0,
        transparent=False,
        bbox_inches="standard",
        pad_inches="inherit",
    )

    with pytest.raises(ValueError, match="padding must be nonnegative"):
        FigureExportState(pad_inches=-0.1)
    with pytest.raises(ValueError, match="dpi must be positive"):
        FigureExportState(dpi=0.0)
    with pytest.raises(ValueError, match="transparency"):
        FigureExportState.model_validate({"transparent": "invalid"})
    with pytest.raises(ValueError, match="bounding box"):
        FigureExportState.model_validate({"bbox_inches": "invalid"})


def test_default_export_padding_accepts_layout(monkeypatch) -> None:
    monkeypatch.setattr(
        figurecomposer_defaults,
        "_styled_rcparams_value",
        lambda _key: "layout",
    )

    assert figurecomposer_defaults._default_export_pad_inches() == "layout"


def test_export_kwargs_follow_style_user_workspace_and_figure_precedence() -> None:
    user_options = AppOptions.model_validate(
        {
            "figure": {
                "export": {
                    "dpi": 180.0,
                    "transparent": "false",
                    "bbox_inches": "standard",
                    "pad_inches": 0.1,
                }
            }
        }
    )
    workspace_options = model_with_workspace_overrides(
        user_options,
        {
            "figure/export/dpi": 240.0,
            "figure/export/bbox_inches": "tight",
        },
    )

    with figure_options_context(workspace_options):
        assert figurecomposer_defaults._resolved_export_kwargs(FigureExportState()) == {
            "dpi": 240.0,
            "transparent": False,
            "bbox_inches": "tight",
            "pad_inches": 0.1,
        }
        assert figurecomposer_defaults._resolved_export_kwargs(
            FigureExportState(
                dpi="figure",
                transparent=True,
                bbox_inches="standard",
                pad_inches="layout",
            )
        ) == {
            "dpi": "figure",
            "transparent": True,
            "bbox_inches": None,
            "pad_inches": "layout",
        }


def test_export_panel_edits_and_resets_per_figure_state(qtbot) -> None:
    panel = FigureExportPanel()
    qtbot.addWidget(panel)
    requested: list[FigureExportState] = []
    panel.state_requested.connect(requested.append)

    panel.dpi_control.mode_combo.setCurrentIndex(
        panel.dpi_control.mode_combo.count() - 1
    )
    panel.dpi_control.value_spin.setValue(320.0)
    panel.transparent_combo.setCurrentIndex(panel.transparent_combo.findData(True))
    panel.bbox_combo.setCurrentIndex(panel.bbox_combo.findData("tight"))
    panel.padding_control.mode_combo.setCurrentIndex(
        panel.padding_control.mode_combo.findData("layout")
    )

    assert requested[-1] == FigureExportState(
        dpi=320.0,
        transparent=True,
        bbox_inches="tight",
        pad_inches="layout",
    )

    panel.use_defaults_button.click()

    assert requested[-1] == FigureExportState()
    assert panel.export_state() == FigureExportState()


def test_export_panel_rejects_an_unknown_control_value(qtbot) -> None:
    panel = FigureExportPanel()
    qtbot.addWidget(panel)

    with pytest.raises(ValueError, match="Unsupported export control value"):
        panel._set_combo_data(panel.bbox_combo, "unknown")


def test_export_only_edits_and_history_do_not_render_or_invalidate_cache(
    qtbot, monkeypatch
) -> None:
    tool = FigureComposerTool(_data())
    qtbot.addWidget(tool)
    tool._reset_history_stack()
    assert tool.editor_tabs.indexOf(tool.export_panel) >= 0

    render_calls: list[object] = []
    monkeypatch.setattr(
        figurecomposer_tool_module,
        "_render_preview",
        lambda owner: render_calls.append(owner),
    )
    info_changes: list[None] = []
    state_changes: list[None] = []
    tool.sigInfoChanged.connect(lambda: info_changes.append(None))
    tool.sigStateChanged.connect(lambda: state_changes.append(None))
    initial_generation = tool.preview_pixmap_generation
    initial_stale = tool.preview_pixmap_stale
    initial_provenance_revision = tool.provenance_revision
    code_before = tool.generated_code()

    export = FigureExportState(
        dpi=300.0,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    tool.export_panel.state_requested.emit(export)

    assert tool.tool_status.export == export
    assert tool.generated_code() == code_before
    assert render_calls == []
    assert info_changes == []
    assert state_changes == [None]
    assert tool.preview_pixmap_generation == initial_generation
    assert tool.preview_pixmap_stale is initial_stale
    assert tool.provenance_revision == initial_provenance_revision
    assert tool.undoable

    tool.undo()

    assert tool.tool_status.export == FigureExportState()
    assert tool.export_panel.export_state() == FigureExportState()
    assert render_calls == []
    assert tool.provenance_revision == initial_provenance_revision

    tool.redo()

    assert tool.tool_status.export == export
    assert tool.export_panel.export_state() == export
    assert render_calls == []
    assert tool.provenance_revision == initial_provenance_revision


def test_export_state_request_ignores_sync_and_unchanged_state(qtbot) -> None:
    tool = FigureComposerTool(_data())
    qtbot.addWidget(tool)
    state_changes: list[None] = []
    tool.sigStateChanged.connect(lambda: state_changes.append(None))

    tool.export_panel.state_requested.emit(FigureExportState())
    tool._updating_controls = True
    tool.export_panel.state_requested.emit(FigureExportState(dpi=300.0))
    tool._updating_controls = False

    assert tool.tool_status.export == FigureExportState()
    assert state_changes == []


def test_explicit_export_resolves_workspace_and_figure_settings(
    qtbot, monkeypatch
) -> None:
    tool = FigureComposerTool(_data())
    qtbot.addWidget(tool)
    user_options = AppOptions.model_validate(
        {
            "figure": {
                "export": {
                    "dpi": 180.0,
                    "transparent": "false",
                    "bbox_inches": "tight",
                    "pad_inches": 0.2,
                }
            }
        }
    )
    effective_options = model_with_workspace_overrides(
        user_options,
        {
            "figure/export/dpi": 240.0,
            "figure/export/transparent": "true",
        },
    )
    tool.set_options_getter(lambda: effective_options)
    tool.export_panel.state_requested.emit(
        FigureExportState(
            dpi="figure",
            bbox_inches="standard",
            pad_inches="layout",
        )
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("figure.png", ""),
    )
    saved: list[tuple[str, dict[str, object]]] = []

    def savefig(_figure: Figure, filename: str, **kwargs: object) -> None:
        saved.append((filename, kwargs))

    monkeypatch.setattr(Figure, "savefig", savefig)

    tool.export_figure()

    assert saved == [
        (
            "figure.png",
            {
                "dpi": "figure",
                "transparent": True,
                "bbox_inches": None,
                "pad_inches": "layout",
            },
        )
    ]


def test_figure_export_state_round_trips_saved_tool(qtbot) -> None:
    tool = FigureComposerTool(_data())
    qtbot.addWidget(tool)
    export = FigureExportState(
        dpi=360.0,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    tool.export_panel.state_requested.emit(export)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(tool.to_dataset())
    assert isinstance(restored, FigureComposerTool)
    qtbot.addWidget(restored)

    assert restored.tool_status.export == export
    assert restored.export_panel.export_state() == export
