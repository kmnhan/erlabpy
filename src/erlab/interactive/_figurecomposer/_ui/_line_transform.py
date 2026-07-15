"""Qt controls for configuring Figure Composer line transforms."""

from __future__ import annotations

import typing

from qtpy import QtWidgets

from erlab.interactive._figurecomposer._line_transform import (
    line_normalize_from_text,
    line_normalize_text,
)
from erlab.interactive._figurecomposer._text import (
    _float_tuple_from_text,
    _format_tuple,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from erlab.interactive._figurecomposer._model._state import FigureOperationState
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


def add_line_transform_controls(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    *,
    object_prefix: str,
    offset_coord_options: Callable[[FigureOperationState], Sequence[str]],
) -> None:
    """Add normalized profile transform controls to an operation editor."""
    normalize_mixed = editor.batch_is_mixed(
        operation, lambda target: target.line_normalize
    )
    normalize_combo = editor.combo(
        [
            line_normalize_text("none"),
            line_normalize_text("max"),
            line_normalize_text("mean"),
        ],
        None if normalize_mixed else line_normalize_text(operation.line_normalize),
        lambda text: editor.request_update(
            line_normalize=line_normalize_from_text(text)
        ),
        parent=page,
        mixed=normalize_mixed,
    )
    normalize_combo.setObjectName(f"{object_prefix}NormalizeCombo")
    editor.add_form_row(
        layout,
        "Normalize",
        normalize_combo,
        "Normalize each extracted 1D profile independently before "
        "scale/offset. This does not normalize the source image or all "
        "profiles together.",
    )

    scales_text, scales_mixed = editor.batch_text(
        operation, lambda target: target.line_scales, _format_tuple
    )
    scales_edit = editor.line_edit(scales_text, parent=page)
    editor.apply_mixed_line_edit(scales_edit, scales_mixed)
    scales_edit.setObjectName(f"{object_prefix}ScalesEdit")
    editor.connect_line_edit_finished(
        scales_edit,
        lambda text: editor.request_update(line_scales=_float_tuple_from_text(text)),
    )
    editor.add_form_row(
        layout,
        "Scales",
        scales_edit,
        "Scale applied to profile data values.\n"
        "Use one value or comma-separated per-profile values.",
    )

    offset_source_mixed = editor.batch_is_mixed(
        operation, lambda target: target.line_offset_source
    )
    offset_source_combo = editor.combo(
        ["manual", "index", "coordinate", "associated"],
        None if offset_source_mixed else operation.line_offset_source,
        lambda source: _update_current_line_offset_source(editor, source),
        parent=page,
        mixed=offset_source_mixed,
    )
    offset_source_combo.setObjectName(f"{object_prefix}OffsetSourceCombo")
    editor.add_form_row(
        layout,
        "Offset source",
        offset_source_combo,
        "Where offsets come from.\n"
        "manual: use Offsets.\n"
        "index/coordinate/associated: derive from profile order or coordinates.",
    )

    if operation.line_offset_source == "associated":
        offset_coord_values = ["", *offset_coord_options(operation)]
        offset_coord_options_match = editor.batch_options_match(
            operation, lambda target: ["", *offset_coord_options(target)]
        )
        offset_coord_mixed = editor.batch_is_mixed(
            operation, lambda target: target.line_offset_coord
        )
        if (
            operation.line_offset_coord is not None
            and operation.line_offset_coord not in offset_coord_values
        ):
            offset_coord_values.append(operation.line_offset_coord)
        offset_coord_combo = editor.combo(
            offset_coord_values,
            None if offset_coord_mixed else operation.line_offset_coord or "",
            lambda text: editor.request_update(line_offset_coord=text or None),
            parent=page,
            mixed=offset_coord_mixed,
            enabled=offset_coord_options_match,
        )
        offset_coord_combo.setObjectName(f"{object_prefix}OffsetCoordinateCombo")
        editor.add_form_row(
            layout,
            "Offset coordinate",
            offset_coord_combo,
            "Associated coordinate used when Offset source is associated."
            + (
                "\nDisabled while selected steps have different valid choices."
                if not offset_coord_options_match
                else ""
            ),
        )

    if operation.line_offset_source != "manual":
        offset_scale_mixed = editor.batch_is_mixed(
            operation, lambda target: target.line_offset_scale
        )
        offset_scale_spin = QtWidgets.QDoubleSpinBox(page)
        offset_scale_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        offset_scale_spin.setDecimals(6)
        offset_scale_spin.setSingleStep(0.1)
        offset_scale_spin.setKeyboardTracking(False)
        offset_scale_spin.setValue(operation.line_offset_scale)
        offset_scale_spin.setObjectName(f"{object_prefix}OffsetScaleEdit")
        editor.connect_value_signal(
            offset_scale_spin,
            offset_scale_spin.valueChanged,
            float,
            lambda value: editor.request_update(line_offset_scale=value),
        )
        offset_scale_tooltip = "Multiplier applied to offsets from the selected source."
        if offset_scale_mixed:
            offset_scale_tooltip += "\nSelected steps have multiple values."
        editor.add_form_row(
            layout,
            "Offset scale",
            editor.mixed_value_widget(
                offset_scale_spin, mixed=offset_scale_mixed, parent=page
            ),
            offset_scale_tooltip,
        )

    if operation.line_offset_source == "manual":
        offsets_text, offsets_mixed = editor.batch_text(
            operation, lambda target: target.line_offsets, _format_tuple
        )
        offsets_edit = editor.line_edit(offsets_text, parent=page)
        editor.apply_mixed_line_edit(offsets_edit, offsets_mixed)
        offsets_edit.setObjectName(f"{object_prefix}OffsetsEdit")
        editor.connect_line_edit_finished(
            offsets_edit,
            lambda text: editor.request_update(
                line_offsets=_float_tuple_from_text(text)
            ),
        )
        editor.add_form_row(
            layout,
            "Offsets",
            offsets_edit,
            "Offset applied to profile data values.\n"
            "Use one value or comma-separated per-profile values.",
        )


def _update_current_line_offset_source(
    editor: FigureOperationEditor, source: str
) -> None:
    updates: dict[str, typing.Any] = {"line_offset_source": source}
    if source == "manual":
        updates["line_offset_scale"] = 1.0
    editor.request_update_rebuild(**updates)
