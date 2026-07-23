"""Operation spec interfaces shared by Figure Composer step modules.

Editor controls that mutate recipe state must use
``FigureOperationEditor.connect_signal`` or an editor factory such as
``combo``/``check_box``. Direct signal connections can fire during widget
population, after a section rebuild, or from a retired Qt wrapper; those signals
must not write recipe state.

Batch-editable controls must also declare a mixed-value presentation:

- text widgets use an empty value with the ``(multiple values)`` placeholder and
  skip commits until the user edits the text;
- combo boxes use the disabled ``(multiple values)`` sentinel item and commit
  only when the user activates a real item;
- check boxes use Qt's partially checked state;
- widgets without native placeholder support, such as spinboxes, sliders, and
  picker buttons, are wrapped with ``FigureOperationEditor.mixed_value_widget`` and
  still connect through the guarded signal helper.

When adding a new operation or method control, implement one of these patterns
before connecting it to recipe updates.
"""

from __future__ import annotations

import dataclasses
import typing

from erlab.interactive._figurecomposer._model._operation_metadata import (
    operation_uses_axes,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._model._state import (
        FigureOperationKind,
        FigureOperationState,
    )
    from erlab.interactive._figurecomposer._tool import FigureComposerTool
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
        StepSection,
    )


@dataclasses.dataclass(frozen=True)
class AddStepActionSpec:
    action_id: str
    text: str
    tooltip: str
    create_operation: Callable[[FigureComposerTool], FigureOperationState]


@dataclasses.dataclass(frozen=True)
class OperationSpec:
    kind: FigureOperationKind
    add_actions: Sequence[AddStepActionSpec]
    display_text: Callable[[FigureComposerTool, FigureOperationState], str]
    tooltip: Callable[[FigureComposerTool, FigureOperationState], str]
    target_text: Callable[[FigureComposerTool, FigureOperationState], str]
    has_invalid_target: Callable[[FigureRecipeContext, FigureOperationState], bool]
    uses_source_section: Callable[[FigureOperationState], bool]
    build_source_editor: Callable[[FigureOperationEditor, FigureOperationState], None]
    build_editor_sections: Callable[
        [FigureOperationEditor, FigureOperationState], Sequence[StepSection]
    ]
    section_summary: Callable[[FigureComposerTool, str, FigureOperationState], str]
    render: Callable[
        [FigureComposerTool, FigureOperationState, Figure, typing.Any], None
    ]
    code_lines: Callable[[FigureComposerTool, FigureOperationState], list[str]]
    render_cache_safe: Callable[[FigureOperationState], bool]
    uses_axes: Callable[[FigureOperationState], bool] = dataclasses.field(
        default=operation_uses_axes
    )
    required_imports: Callable[
        [FigureComposerTool, FigureOperationState], Sequence[str]
    ] = dataclasses.field(default=lambda _tool, _operation: ())
    loaded_operation: Callable[[FigureOperationState], FigureOperationState] = (
        dataclasses.field(default=lambda operation: operation)
    )


def _empty_source_editor(
    _editor: FigureOperationEditor, _operation: FigureOperationState
) -> None:
    return


def _empty_code_lines(
    _tool: FigureComposerTool, _operation: FigureOperationState
) -> list[str]:
    return []


def _no_invalid_target(
    _context: FigureRecipeContext, _operation: FigureOperationState
) -> bool:
    return False


def _uses_no_source_section(_operation: FigureOperationState) -> bool:
    return False


def _always_render_cache_safe(_operation: FigureOperationState) -> bool:
    """Declare that an operation cannot mutate Figure Composer source data."""
    return True


def _empty_section_summary(
    _tool: FigureComposerTool, _key: str, _operation: FigureOperationState
) -> str:
    return ""
