"""Private exceptions for Figure Composer launch and recipe construction."""

from __future__ import annotations

PLOT_SLICES_SELECTION_ERROR_TITLE = "Cannot Create Plot Slices Figure"
PLOT_SLICES_SELECTION_ERROR_MESSAGE = (
    "The current selection cannot be represented as an editable plot_slices "
    "coordinate selection.\n\n"
    "Prepare the data first by sorting coordinates with DataArray.sortby, "
    "removing duplicate coordinates, interpolating to suitable coordinates, or "
    "opening a new ImageTool containing only the data you want to plot."
)


class FigureComposerSelectionError(ValueError):
    """Base class for Figure Composer selection construction failures."""


class FigureComposerPlotSlicesSelectionError(FigureComposerSelectionError):
    """Raised when an ImageTool pane cannot seed editable plot_slices selection."""

    def __init__(self, detail: str | None = None) -> None:
        message = PLOT_SLICES_SELECTION_ERROR_MESSAGE
        if detail:
            message = f"{message}\n\nDetails: {detail}"
        super().__init__(message)
