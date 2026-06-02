"""Private Figure Composer framework used by ImageTool manager."""

from __future__ import annotations

from erlab.interactive._figurecomposer._norms import _norm_updates_from_kwargs
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureDataSelectionState,
    FigureExportState,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecLayoutState,
    FigureGridSpecSpanState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._tool import FigureComposerTool

__all__ = [
    "FigureAxesSelectionState",
    "FigureComposerTool",
    "FigureDataSelectionState",
    "FigureExportState",
    "FigureGridSpecAxesState",
    "FigureGridSpecGridState",
    "FigureGridSpecLayoutState",
    "FigureGridSpecSpanState",
    "FigureMethodFamily",
    "FigureOperationKind",
    "FigureOperationState",
    "FigureRecipeState",
    "FigureSourceState",
    "FigureSubplotsState",
    "_norm_updates_from_kwargs",
]
