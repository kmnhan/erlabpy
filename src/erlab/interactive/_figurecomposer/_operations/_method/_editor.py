"""Qt editor for curated Figure Composer method operations."""

from __future__ import annotations

import contextlib
import functools
import typing

from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive._figurecomposer._line_style import (
    color_kw_value_from_text,
    normalize_style_value,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._method._catalog import (
    _CALL_POLICY_LABELS,
    _FAMILY_LABELS,
    _FLOAT_SPINBOX_DECIMALS,
    _FLOAT_SPINBOX_MAXIMUM,
    _FLOAT_SPINBOX_MINIMUM,
    _FLOAT_SPINBOX_STEP,
    _INT_SPINBOX_MAXIMUM,
    _INT_SPINBOX_MINIMUM,
    _TRANSFORM_COMPONENT_OPTIONS,
    TICK_PARAMS_CONTROLLED_KWARGS,
    MethodCallPolicy,
    MethodControlKind,
    MethodControlSpec,
    MethodSpec,
    MethodTargetDomain,
    MethodTextValuesPolicy,
    _callable_display,
    _effective_call_policy,
    _method_display,
    _method_doc_url,
    _method_selector_text,
    _method_spec,
    _method_specs,
)
from erlab.interactive._figurecomposer._operations._method._plot_editor import (
    _build_plot_data_args_editor,
)
from erlab.interactive._figurecomposer._operations._method._state import (
    _aspect_value_from_text,
    _default_method_args,
    _empty_text_as_none,
    _format_aspect_value,
    _format_float_value,
    _format_int_value,
    _format_literal_value,
    _is_axes_errorbar_method,
    _is_layout_engine_method,
    _layout_engine_kwarg_keys,
    _layout_engine_name,
    _literal_value_from_text,
    _method_arg_value,
    _method_args,
    _method_has_transform_control,
    _method_kwarg_value,
    _method_transfer_updates,
    _optional_float_from_text,
    _optional_int_from_text,
    _optional_literal_from_text,
    _string_tuple_from_text_or_none,
    _subplots_adjust_values,
)
from erlab.interactive._figurecomposer._subplot_adjust import (
    SUBPLOTS_ADJUST_SPINBOX_DECIMALS,
    SUBPLOTS_ADJUST_SPINBOX_STEP,
    normalize_subplots_adjust_kwargs,
    subplots_adjust_spinbox_range,
)
from erlab.interactive._figurecomposer._text import (
    _dict_from_text,
    _float_pair_from_text,
    _format_dict,
    _format_limit_pair,
    _format_literal_sequence,
    _format_pair,
    _format_string_tuple,
    _limit_pair_from_text,
    _limit_pair_from_value,
    _literal_sequence_from_text,
    _string_tuple_from_text,
    _text_tuple_from_text,
)
from erlab.interactive._figurecomposer._ui._color_widgets import _ColorLineEditWidget
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection
from erlab.interactive._figurecomposer._ui._tick_params import TickParamsEditorWidget

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


def _method_combo(
    editor: FigureOperationEditor,
    family: FigureMethodFamily,
    current_name: str,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox(parent)
    editor.mark_control(combo)
    for spec in _method_specs(family).values():
        combo.addItem(_method_selector_text(spec), spec.name)
        combo.setItemData(
            combo.count() - 1,
            spec.tooltip,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
    for index in range(combo.count()):
        if combo.itemData(index) == current_name:
            combo.setCurrentIndex(index)
            break

    def method_activated(_index: int) -> None:
        method_name = combo.currentData()
        if isinstance(method_name, str):
            _update_current_method_name(editor, method_name)

    editor.connect_signal(combo, combo.activated, method_activated)
    return combo


def _method_plain_text_edit(
    editor: FigureOperationEditor,
    text: str,
    *,
    mixed: bool,
    object_name: str,
    parent: QtWidgets.QWidget | None,
) -> QtWidgets.QPlainTextEdit:
    edit = QtWidgets.QPlainTextEdit(parent)
    edit.setPlainText(text)
    editor.apply_mixed_plain_text_edit(edit, mixed)
    edit.setMaximumHeight(70)
    edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    edit.setObjectName(object_name)
    return edit


def _build_method_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> Sequence[StepSection]:
    page, layout = editor.new_form_page("figureComposerMethodPage")
    spec = _method_spec(operation)

    editor.add_form_section(
        layout,
        "Command",
        object_name="figureComposerMethodCallSection",
    )
    family_combo = editor.combo(
        [label for _family, label in _FAMILY_LABELS.items()],
        _FAMILY_LABELS[operation.method_family],
        lambda text: _update_current_method_family(editor, _family_from_label(text)),
        parent=page,
    )
    family_combo.setObjectName("figureComposerMethodFamilyCombo")
    editor.add_form_row(
        layout,
        "Family",
        family_combo,
        (
            "Choose whether this step calls an ERLab helper, "
            "an Axes method, or a Figure method."
        ),
    )

    method_widget = QtWidgets.QWidget(page)
    method_layout = QtWidgets.QHBoxLayout(method_widget)
    method_layout.setContentsMargins(0, 0, 0, 0)
    method_layout.setSpacing(6)
    method_combo = _method_combo(
        editor, operation.method_family, spec.name, method_widget
    )
    method_combo.setObjectName(_method_combo_object_name(operation.method_family))
    method_combo.setToolTip("Function or method called by this recipe step.")
    method_layout.addWidget(method_combo, 1)
    docs_button = QtWidgets.QToolButton(method_widget)
    docs_button.setObjectName("figureComposerMethodDocsButton")
    docs_button.setText("Docs")
    docs_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
    docs_button.setToolTip("Open API documentation for this method.")
    doc_url = _method_doc_url(spec)
    docs_button.setProperty("figure_method_doc_url", doc_url or "")
    docs_button.setEnabled(doc_url is not None)
    if doc_url is not None:
        editor.connect_signal(
            docs_button,
            docs_button.clicked,
            lambda _checked=False, doc_url=doc_url: _open_method_doc_url(doc_url),
        )
    method_layout.addWidget(docs_button)
    editor.add_form_row(
        layout,
        "Method",
        method_widget,
        "Function or method called by this recipe step.",
    )

    if len(spec.selectable_call_policies) > 1:
        policy = _effective_call_policy(operation, spec)
        policy_mixed = editor.batch_is_mixed(
            operation, lambda target: _effective_call_policy(target, spec)
        )
        policy_combo = editor.combo(
            [
                _CALL_POLICY_LABELS.get(item, item.value)
                for item in spec.selectable_call_policies
            ],
            None if policy_mixed else _CALL_POLICY_LABELS.get(policy, policy.value),
            lambda text: _update_current_method_call_policy(
                editor, _call_policy_from_label(text)
            ),
            parent=page,
            mixed=policy_mixed,
        )
        policy_combo.setObjectName("figureComposerMethodCallPolicyCombo")
        editor.add_form_row(
            layout,
            "Apply to",
            policy_combo,
            (
                "Choose whether this method receives all selected axes "
                "at once or runs once per axis."
            ),
        )

    has_value_controls = (
        bool(spec.controls) or spec.text_values_policy != MethodTextValuesPolicy.NONE
    )
    if has_value_controls:
        editor.add_form_section(
            layout,
            "Parameters",
            object_name="figureComposerMethodValuesSection",
        )

    if spec.text_values_policy != MethodTextValuesPolicy.NONE:
        text_values_text, text_values_mixed = editor.batch_text(
            operation,
            lambda target: target.text_values,
            lambda value: "\n".join(typing.cast("Sequence[str]", value)),
        )
        text_edit = QtWidgets.QPlainTextEdit(page)
        text_edit.setPlainText(text_values_text)
        editor.apply_mixed_plain_text_edit(text_edit, text_values_mixed)
        text_edit.setMaximumHeight(70)
        text_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_edit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_edit.setObjectName("figureComposerMethodTextValuesEdit")
        editor.connect_plain_text_changed(
            text_edit,
            lambda text: _update_current_method_text_values(editor, text),
        )
        editor.add_form_row(
            layout,
            "Text",
            text_edit,
            "One text value per line for methods that apply labels or annotations.",
        )

    for control in spec.controls:
        if _method_control_visible(operation, spec, control):
            _add_method_control_row(editor, layout, operation, spec, control)

    if not spec.controls and spec.text_values_policy == MethodTextValuesPolicy.NONE:
        label = QtWidgets.QLabel(f"{_callable_display(spec)} takes no values.", page)
        label.setWordWrap(True)
        editor.add_form_row(
            layout,
            "Action",
            label,
            "This method runs directly on its configured target.",
        )

    if spec.allow_extra_kwargs:
        if has_value_controls:
            editor.add_form_section(
                layout,
                "Advanced",
                object_name="figureComposerMethodAdvancedSection",
            )
        kwargs_text, kwargs_mixed = editor.batch_text(
            operation,
            lambda target: _extra_method_kwargs(target, spec),
            lambda value: _format_dict(typing.cast("dict[str, typing.Any]", value)),
        )
        kwargs_edit = editor.line_edit(kwargs_text, parent=page)
        editor.apply_mixed_line_edit(kwargs_edit, kwargs_mixed)
        kwargs_edit.setObjectName(_method_kwargs_object_name(operation.method_family))
        editor.connect_line_edit_finished(
            kwargs_edit,
            lambda text: _update_current_extra_method_kwargs(
                editor, spec, _dict_from_text(text)
            ),
        )
        editor.add_form_row(
            layout,
            "Extra kwargs",
            kwargs_edit,
            f"Keyword arguments forwarded to {_callable_display(spec)}.",
        )

    return (
        StepSection(
            "method",
            _method_display(operation),
            page,
            "Configure the curated function or method call for this step.",
        ),
    )


def _add_method_control_row(
    editor: FigureOperationEditor,
    layout: QtWidgets.QFormLayout,
    operation: FigureOperationState,
    spec: MethodSpec,
    control: MethodControlSpec,
) -> None:
    match control.kind:
        case MethodControlKind.TRANSFORM:
            mode_mixed = editor.batch_is_mixed(
                operation, lambda target: target.method_transform
            )
            mode_combo = editor.combo(
                control.options,
                None if mode_mixed else operation.method_transform,
                _method_transform_update_callback(editor),
                parent=layout.parentWidget(),
                mixed=mode_mixed,
            )
            mode_combo.setObjectName(control.object_name)
            editor.add_form_row(layout, control.label, mode_combo, control.tooltip)
            if not mode_mixed and operation.method_transform == "blend":
                x_mixed = editor.batch_is_mixed(
                    operation, lambda target: target.method_transform_x
                )
                x_combo = editor.combo(
                    _TRANSFORM_COMPONENT_OPTIONS,
                    None if x_mixed else operation.method_transform_x,
                    lambda text: editor.request_update(method_transform_x=text),
                    parent=layout.parentWidget(),
                    mixed=x_mixed,
                )
                x_combo.setObjectName("figureComposerMethodTransformXCombo")
                y_mixed = editor.batch_is_mixed(
                    operation, lambda target: target.method_transform_y
                )
                y_combo = editor.combo(
                    _TRANSFORM_COMPONENT_OPTIONS,
                    None if y_mixed else operation.method_transform_y,
                    lambda text: editor.request_update(method_transform_y=text),
                    parent=layout.parentWidget(),
                    mixed=y_mixed,
                )
                y_combo.setObjectName("figureComposerMethodTransformYCombo")
                editor.add_compound_form_row(
                    layout,
                    "Blend",
                    (
                        ("x", x_combo, "Transform used for x coordinates."),
                        ("y", y_combo, "Transform used for y coordinates."),
                    ),
                    "Build a blended transform from separate x and y components.",
                )
            elif not mode_mixed and operation.method_transform == "custom":
                expression_text, expression_mixed = editor.batch_text(
                    operation, lambda target: target.method_transform_expression, str
                )
                expression_edit = editor.line_edit(
                    expression_text,
                    parent=layout.parentWidget(),
                )
                editor.apply_mixed_line_edit(expression_edit, expression_mixed)
                expression_edit.setObjectName(
                    "figureComposerMethodTransformExpressionEdit"
                )
                editor.connect_line_edit_finished(
                    expression_edit,
                    lambda text: editor.request_update(
                        method_transform_expression=text
                    ),
                )
                editor.add_form_row(
                    layout,
                    "Expression",
                    expression_edit,
                    "Python expression for transform=.\n"
                    "Available names: ax, fig, mtransforms.",
                )
                trusted_check = editor.check_box(
                    operation.trusted,
                    _operation_trust_update_callback(editor),
                    parent=layout.parentWidget(),
                )
                trusted_check.setObjectName("figureComposerMethodTransformTrustedCheck")
                editor.add_form_row(
                    layout,
                    "Trusted",
                    trusted_check,
                    "Allow this custom transform expression to execute.",
                )
        case MethodControlKind.ARG_COMBO:
            index = _control_arg_index(control)
            arg_value_getter: Callable[[FigureOperationState], typing.Any]
            if _is_layout_engine_method(spec):

                def arg_value_getter(target: FigureOperationState) -> typing.Any:
                    return _layout_engine_name(target, spec)
            else:

                def arg_value_getter(target: FigureOperationState) -> typing.Any:
                    return _method_arg_value(target, spec, index, control.default)

            mixed = editor.batch_is_mixed(
                operation,
                arg_value_getter,
            )
            combo = editor.combo(
                control.options,
                None if mixed else str(arg_value_getter(operation)),
                _method_arg_callback(editor, index, spec),
                parent=layout.parentWidget(),
                mixed=mixed,
            )
            combo.setObjectName(control.object_name)
            editor.add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.INT_ARG:
            index = _control_arg_index(control)
            mixed = editor.batch_is_mixed(
                operation,
                lambda target: _method_arg_value(target, spec, index, control.default),
            )
            spinbox = _int_spinbox(
                control.default
                if mixed
                else _method_arg_value(operation, spec, index, control.default),
                control,
                parent=layout.parentWidget(),
            )
            spinbox.setObjectName(control.object_name)
            editor.connect_value_signal(
                spinbox,
                spinbox.valueChanged,
                int,
                _method_arg_update_callback(editor, index),
            )
            editor.add_form_row(
                layout,
                control.label,
                editor.mixed_value_widget(
                    spinbox, mixed=mixed, parent=layout.parentWidget()
                ),
                _numeric_control_tooltip(control, mixed),
            )
        case MethodControlKind.FLOAT_ARG:
            index = _control_arg_index(control)
            mixed = editor.batch_is_mixed(
                operation,
                lambda target: _method_arg_value(target, spec, index, 0.0),
            )
            spinbox = _float_spinbox(
                control.default
                if mixed
                else _method_arg_value(operation, spec, index, 0.0),
                control,
                parent=layout.parentWidget(),
            )
            spinbox.setObjectName(control.object_name)
            editor.connect_value_signal(
                spinbox,
                spinbox.valueChanged,
                float,
                _method_arg_update_callback(editor, index),
            )
            editor.add_form_row(
                layout,
                control.label,
                editor.mixed_value_widget(
                    spinbox, mixed=mixed, parent=layout.parentWidget()
                ),
                _numeric_control_tooltip(control, mixed),
            )
        case MethodControlKind.TEXT_ARG:
            index = _control_arg_index(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, ""),
                str,
            )
            edit = _method_plain_text_edit(
                editor,
                text,
                mixed=mixed,
                object_name=control.object_name,
                parent=layout.parentWidget(),
            )
            editor.connect_plain_text_changed(
                edit,
                _method_arg_update_callback(editor, index),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.LITERAL_ARG:
            index = _control_arg_index(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, control.default),
                _format_literal_value,
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _parsed_method_arg_update_callback(
                    editor,
                    index,
                    _literal_value_from_text,
                ),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.LITERAL_SEQUENCE_ARG:
            index = _control_arg_index(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, ()),
                lambda value: (
                    _format_literal_sequence(typing.cast("Sequence[typing.Any]", value))
                    if isinstance(value, (list, tuple))
                    else repr(value)
                ),
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _parsed_method_arg_update_callback(
                    editor,
                    index,
                    _literal_sequence_from_text,
                ),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.PLOT_DATA_ARGS:
            _build_plot_data_args_editor(editor, operation, spec, layout)
        case MethodControlKind.TICK_PARAMS:
            _build_tick_params_editor(editor, operation, layout)
        case MethodControlKind.STRING_TUPLE_ARG:
            index = _control_arg_index(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, ()),
                lambda value: (
                    _format_string_tuple(typing.cast("Sequence[str]", value))
                    if isinstance(value, (list, tuple))
                    else ""
                ),
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _method_string_tuple_arg_update_callback(editor, index),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_PAIR_ARGS:
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_float_pair_args(editor, target, spec),
                _format_limit_pair,
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _method_float_pair_args_update_callback(editor),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.ASPECT_ARG:
            index = _control_arg_index(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, control.default),
                _format_aspect_value,
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _parsed_method_arg_update_callback(
                    editor,
                    index,
                    _aspect_value_from_text,
                ),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.BOOL_ARG_COMBO:
            index = _control_arg_index(control)
            mixed = editor.batch_is_mixed(
                operation,
                lambda target: bool(
                    _method_arg_value(target, spec, index, control.default)
                ),
            )
            combo = editor.combo(
                control.options,
                None
                if mixed
                else str(
                    bool(_method_arg_value(operation, spec, index, control.default))
                ),
                _method_bool_arg_callback(editor, index),
                parent=layout.parentWidget(),
                mixed=mixed,
            )
            combo.setObjectName(control.object_name)
            editor.add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.KWARG_COMBO:
            key = _control_key(control)
            kwarg_value_getter: Callable[[FigureOperationState], typing.Any]
            if control.none_label is None:

                def kwarg_value_getter(target: FigureOperationState) -> typing.Any:
                    return _method_kwarg_value(target, key, control.default)

            else:

                def kwarg_value_getter(target: FigureOperationState) -> typing.Any:
                    return normalize_style_value(
                        _method_kwarg_value(target, key, control.default)
                    )

            mixed = editor.batch_is_mixed(
                operation,
                kwarg_value_getter,
            )
            if control.none_label is None:
                combo = editor.combo(
                    control.options,
                    None if mixed else str(kwarg_value_getter(operation)),
                    _method_kwarg_callback(editor, key),
                    parent=layout.parentWidget(),
                    mixed=mixed,
                )
            else:
                combo = editor.optional_name_combo(
                    control.options,
                    None if mixed else kwarg_value_getter(operation),
                    control.none_label,
                    _method_optional_kwarg_callback(editor, key),
                    parent=layout.parentWidget(),
                    mixed=mixed,
                )
            combo.setObjectName(control.object_name)
            editor.add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.BOOL_KWARG_COMBO:
            key = _control_key(control)
            mixed = editor.batch_is_mixed(
                operation,
                lambda target: bool(_method_kwarg_value(target, key, control.default)),
            )
            combo = editor.combo(
                control.options,
                None
                if mixed
                else str(bool(_method_kwarg_value(operation, key, control.default))),
                _method_bool_kwarg_callback(editor, key),
                parent=layout.parentWidget(),
                mixed=mixed,
            )
            combo.setObjectName(control.object_name)
            editor.add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.OPTIONAL_BOOL_KWARG_COMBO:
            key = _control_key(control)

            def kwarg_value_getter(target: FigureOperationState) -> bool | None:
                value = _method_kwarg_value(target, key, control.default)
                return value if isinstance(value, bool) else None

            mixed = editor.batch_is_mixed(
                operation,
                kwarg_value_getter,
            )
            value = kwarg_value_getter(operation)
            combo = editor.optional_name_combo(
                control.options,
                None if mixed or value is None else str(value),
                control.none_label or "Default",
                _method_optional_bool_kwarg_callback(editor, key),
                parent=layout.parentWidget(),
                mixed=mixed,
            )
            combo.setObjectName(control.object_name)
            editor.add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.INT_KWARG:
            key = _control_key(control)
            if _control_uses_numeric_spinbox(control):
                mixed = editor.batch_is_mixed(
                    operation,
                    lambda target: _method_kwarg_value(target, key, control.default),
                )
                spinbox = _int_spinbox(
                    control.default
                    if mixed
                    else _method_kwarg_value(operation, key, control.default),
                    control,
                    parent=layout.parentWidget(),
                )
                spinbox.setObjectName(control.object_name)
                editor.connect_value_signal(
                    spinbox,
                    spinbox.valueChanged,
                    int,
                    _method_kwarg_update_callback(editor, key),
                )
                editor.add_form_row(
                    layout,
                    control.label,
                    editor.mixed_value_widget(
                        spinbox, mixed=mixed, parent=layout.parentWidget()
                    ),
                    _numeric_control_tooltip(control, mixed),
                )
            else:
                text, mixed = editor.batch_text(
                    operation,
                    lambda target: _method_kwarg_value(target, key, control.default),
                    _format_int_value,
                )
                edit = editor.line_edit(text, parent=layout.parentWidget())
                editor.apply_mixed_line_edit(edit, mixed)
                edit.setObjectName(control.object_name)
                editor.connect_line_edit_finished(
                    edit,
                    _parsed_method_kwarg_update_callback(
                        editor,
                        key,
                        _optional_int_from_text,
                    ),
                )
                editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_KWARG:
            key = _control_key(control)
            if _control_uses_numeric_spinbox(control):
                mixed = editor.batch_is_mixed(
                    operation,
                    lambda target: _method_kwarg_value(target, key, control.default),
                )
                spinbox = _float_spinbox(
                    control.default
                    if mixed
                    else _method_kwarg_value(operation, key, control.default),
                    control,
                    parent=layout.parentWidget(),
                )
                spinbox.setObjectName(control.object_name)
                editor.connect_value_signal(
                    spinbox,
                    spinbox.valueChanged,
                    float,
                    _method_kwarg_update_callback(editor, key),
                )
                editor.add_form_row(
                    layout,
                    control.label,
                    editor.mixed_value_widget(
                        spinbox, mixed=mixed, parent=layout.parentWidget()
                    ),
                    _numeric_control_tooltip(control, mixed),
                )
            else:
                text, mixed = editor.batch_text(
                    operation,
                    lambda target: _method_kwarg_value(target, key, control.default),
                    _format_float_value,
                )
                edit = editor.line_edit(text, parent=layout.parentWidget())
                editor.apply_mixed_line_edit(edit, mixed)
                edit.setObjectName(control.object_name)
                editor.connect_line_edit_finished(
                    edit,
                    _parsed_method_kwarg_update_callback(
                        editor,
                        key,
                        _optional_float_from_text,
                    ),
                )
                editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.SUBPLOTS_ADJUST_KWARG:
            key = _control_key(control)
            mixed = editor.batch_is_mixed(
                operation,
                lambda target: _method_kwarg_value(
                    target, key, editor.subplot_parameter_default(key)
                ),
            )
            spinbox = _subplots_adjust_spinbox(
                editor,
                operation,
                key,
                mixed=mixed,
                parent=layout.parentWidget(),
            )
            spinbox.setObjectName(control.object_name)
            tooltip = control.tooltip
            if mixed:
                tooltip += "\nSelected steps have multiple values."
            editor.add_form_row(
                layout,
                control.label,
                editor.mixed_value_widget(
                    spinbox, mixed=mixed, parent=layout.parentWidget()
                ),
                tooltip,
            )
        case MethodControlKind.TEXT_KWARG:
            key = _control_key(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, control.default),
                lambda value: str(value or ""),
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(editor, key, _empty_text_as_none),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.COLOR_KWARG:
            key = _control_key(control)
            color_text, color_mixed = editor.batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, ""),
                lambda value: str(value or ""),
            )
            color_edit = _ColorLineEditWidget(
                color_text,
                parent=layout.parentWidget(),
                inherited_color=(
                    _method_kwarg_value(operation, "color", None)
                    if key != "color"
                    else None
                ),
            )
            color_edit.setLineEditObjectName(control.object_name)
            color_edit.setColorButtonObjectName(f"{control.object_name}Button")
            editor.apply_mixed_line_edit(color_edit.line_edit, color_mixed)
            editor.connect_value_signal(
                color_edit,
                color_edit.editingFinished,
                color_edit.text,
                _method_color_kwarg_update_callback(editor, key),
                unchanged_mixed=lambda: editor.line_edit_batch_unchanged(
                    color_edit.line_edit
                ),
            )
            editor.add_form_row(layout, control.label, color_edit, control.tooltip)
        case MethodControlKind.LITERAL_KWARG:
            key = _control_key(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, control.default),
                _format_literal_value,
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(
                    editor,
                    key,
                    _optional_literal_from_text,
                ),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.STRING_TUPLE_KWARG:
            key = _control_key(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, ()),
                lambda value: (
                    _format_string_tuple(typing.cast("Sequence[str]", value))
                    if isinstance(value, (list, tuple))
                    else ""
                ),
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(
                    editor,
                    key,
                    _string_tuple_from_text_or_none,
                ),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_PAIR_KWARG:
            key = _control_key(control)
            text, mixed = editor.batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, control.default),
                _format_pair,
            )
            edit = editor.line_edit(text, parent=layout.parentWidget())
            editor.apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            editor.connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(
                    editor,
                    key,
                    _float_pair_from_text,
                ),
            )
            editor.add_form_row(layout, control.label, edit, control.tooltip)


def _method_arg_update_callback(
    editor: FigureOperationEditor, index: int
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_method_arg, editor, index)


def _parsed_method_arg_update_callback(
    editor: FigureOperationEditor,
    index: int,
    parser: Callable[[typing.Any], typing.Any],
) -> Callable[[typing.Any], None]:
    return functools.partial(
        _update_current_method_arg_from_value, editor, index, parser
    )


def _method_kwarg_update_callback(
    editor: FigureOperationEditor, key: str
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_method_kwarg, editor, key)


def _subplots_adjust_kwarg_update_callback(
    editor: FigureOperationEditor, key: str
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_subplots_adjust_kwarg, editor, key)


def _method_color_kwarg_update_callback(
    editor: FigureOperationEditor, key: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_kwarg(editor, key, color_kw_value_from_text(text))

    return update


def _method_transform_update_callback(
    editor: FigureOperationEditor,
) -> Callable[[str], None]:
    def update(text: str) -> None:
        editor.request_update_rebuild(
            method_transform=text,
            trusted=text == "custom",
        )

    return update


def _operation_trust_update_callback(
    editor: FigureOperationEditor,
) -> Callable[[bool], None]:
    def update(checked: bool) -> None:
        editor.request_update(trusted=checked)

    return update


def _parsed_method_kwarg_update_callback(
    editor: FigureOperationEditor,
    key: str,
    parser: Callable[[typing.Any], typing.Any],
) -> Callable[[typing.Any], None]:
    return functools.partial(
        _update_current_method_kwarg_from_value, editor, key, parser
    )


def _method_string_tuple_arg_update_callback(
    editor: FigureOperationEditor, index: int
) -> Callable[[str], None]:
    return functools.partial(_update_current_method_string_tuple_arg, editor, index)


def _method_float_pair_args_update_callback(
    editor: FigureOperationEditor,
) -> Callable[[str], None]:
    return functools.partial(_update_current_method_args_from_pair_text, editor)


def _update_current_method_arg_from_value(
    editor: FigureOperationEditor,
    index: int,
    parser: Callable[[typing.Any], typing.Any],
    value: typing.Any,
) -> None:
    _update_current_method_arg(editor, index, parser(value))


def _update_current_method_kwarg_from_value(
    editor: FigureOperationEditor,
    key: str,
    parser: Callable[[typing.Any], typing.Any],
    value: typing.Any,
) -> None:
    _update_current_method_kwarg(editor, key, parser(value))


def _update_current_method_args_from_pair_text(
    editor: FigureOperationEditor, text: str
) -> None:
    _update_current_method_args(editor, _limit_pair_from_text(text) or ())


def _method_float_pair_args(
    editor: FigureOperationEditor, operation: FigureOperationState, spec: MethodSpec
) -> tuple[float | None, float | None] | None:
    args = _method_args(
        operation,
        spec,
        default_args=_default_method_args(spec, editor.first_live_axis(operation.axes)),
    )
    if len(args) < 2:
        return None
    return _limit_pair_from_value(args[:2])


def _controlled_method_kwarg_keys(spec: MethodSpec) -> frozenset[str]:
    keys = {
        control.key
        for control in spec.controls
        if control.key is not None
        and control.kind
        in {
            MethodControlKind.KWARG_COMBO,
            MethodControlKind.BOOL_KWARG_COMBO,
            MethodControlKind.OPTIONAL_BOOL_KWARG_COMBO,
            MethodControlKind.INT_KWARG,
            MethodControlKind.FLOAT_KWARG,
            MethodControlKind.SUBPLOTS_ADJUST_KWARG,
            MethodControlKind.TEXT_KWARG,
            MethodControlKind.LITERAL_KWARG,
            MethodControlKind.STRING_TUPLE_KWARG,
            MethodControlKind.FLOAT_PAIR_KWARG,
            MethodControlKind.COLOR_KWARG,
        }
    }
    if any(control.kind == MethodControlKind.TICK_PARAMS for control in spec.controls):
        keys.update(TICK_PARAMS_CONTROLLED_KWARGS)
    if _is_axes_errorbar_method(spec):
        keys.update(("xerr", "yerr"))
    if _method_has_transform_control(spec):
        keys.add("transform")
    return frozenset(keys)


def _extra_method_kwargs(
    operation: FigureOperationState, spec: MethodSpec
) -> dict[str, typing.Any]:
    controlled = _controlled_method_kwarg_keys(spec)
    return {
        key: value
        for key, value in operation.method_kwargs.items()
        if key not in controlled
    }


def _update_current_extra_method_kwargs(
    editor: FigureOperationEditor, spec: MethodSpec, extra_kwargs: dict[str, typing.Any]
) -> None:
    controlled = _controlled_method_kwarg_keys(spec)

    def update_kwargs(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        kwargs = {
            key: value
            for key, value in operation.method_kwargs.items()
            if key in controlled
        }
        kwargs.update(
            {key: value for key, value in extra_kwargs.items() if key not in controlled}
        )
        return operation.model_copy(update={"method_kwargs": kwargs})

    editor.request_transform(update_kwargs)


def _update_current_tick_params_kwargs(
    editor: FigureOperationEditor, tick_kwargs: Mapping[str, typing.Any]
) -> None:
    def update_kwargs(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        kwargs = {
            key: value
            for key, value in operation.method_kwargs.items()
            if key not in TICK_PARAMS_CONTROLLED_KWARGS
        }
        kwargs.update(
            {
                key: value
                for key, value in tick_kwargs.items()
                if key in TICK_PARAMS_CONTROLLED_KWARGS
            }
        )
        return operation.model_copy(update={"method_kwargs": kwargs})

    editor.request_transform(update_kwargs)


def _build_tick_params_editor(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    layout: QtWidgets.QFormLayout,
) -> None:
    tick_kwargs = {
        key: value
        for key, value in operation.method_kwargs.items()
        if key in TICK_PARAMS_CONTROLLED_KWARGS
    }
    mixed = editor.batch_is_mixed(
        operation,
        lambda target: {
            key: value
            for key, value in target.method_kwargs.items()
            if key in TICK_PARAMS_CONTROLLED_KWARGS
        },
    )
    tick_editor = TickParamsEditorWidget(
        {} if mixed else tick_kwargs,
        parent=layout.parentWidget(),
    )
    editor.connect_value_signal(
        tick_editor,
        tick_editor.sigTickParamsChanged,
        lambda kwargs: dict(kwargs),
        lambda kwargs: _update_current_tick_params_kwargs(editor, kwargs),
    )
    editor.add_form_row(
        layout,
        "Ticks",
        editor.mixed_value_widget(
            tick_editor, mixed=mixed, parent=layout.parentWidget()
        ),
        "Compact controls for ax.tick_params.",
    )


def _control_uses_numeric_spinbox(control: MethodControlSpec) -> bool:
    return control.default is not None


def _numeric_control_tooltip(control: MethodControlSpec, mixed: bool) -> str:
    if not mixed:
        return control.tooltip
    return f"{control.tooltip}\nSelected steps have multiple values."


def _int_spinbox(
    value: typing.Any,
    control: MethodControlSpec,
    *,
    parent: QtWidgets.QWidget | None,
) -> QtWidgets.QSpinBox:
    spinbox = QtWidgets.QSpinBox(parent)
    spinbox.setRange(
        _INT_SPINBOX_MINIMUM if control.minimum is None else int(control.minimum),
        _INT_SPINBOX_MAXIMUM if control.maximum is None else int(control.maximum),
    )
    spinbox.setSingleStep(1 if control.step is None else int(control.step))
    spinbox.setKeyboardTracking(False)
    if value is None:
        value = 0 if control.default is None else control.default
    spinbox.setValue(int(value))
    return spinbox


def _float_spinbox(
    value: typing.Any,
    control: MethodControlSpec,
    *,
    parent: QtWidgets.QWidget | None,
) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox(parent)
    spinbox.setDecimals(
        _FLOAT_SPINBOX_DECIMALS if control.decimals is None else control.decimals
    )
    spinbox.setRange(
        _FLOAT_SPINBOX_MINIMUM if control.minimum is None else float(control.minimum),
        _FLOAT_SPINBOX_MAXIMUM if control.maximum is None else float(control.maximum),
    )
    spinbox.setSingleStep(
        _FLOAT_SPINBOX_STEP if control.step is None else float(control.step)
    )
    spinbox.setKeyboardTracking(False)
    if value is None:
        value = 0.0 if control.default is None else control.default
    spinbox.setValue(float(value))
    return spinbox


def _subplots_adjust_spinbox(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    key: str,
    *,
    mixed: bool,
    parent: QtWidgets.QWidget | None,
) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox(parent)
    defaults = {
        name: editor.subplot_parameter_default(name)
        for name in ("left", "bottom", "right", "top", "wspace", "hspace")
    }
    values = _subplots_adjust_values(operation, defaults)
    minimum, maximum = subplots_adjust_spinbox_range(key, values)
    spinbox.setRange(minimum, maximum)
    spinbox.setDecimals(SUBPLOTS_ADJUST_SPINBOX_DECIMALS)
    spinbox.setSingleStep(SUBPLOTS_ADJUST_SPINBOX_STEP)
    spinbox.setKeyboardTracking(False)
    value = float(values[key])
    if not mixed:
        with contextlib.suppress(TypeError, ValueError):
            value = float(_method_kwarg_value(operation, key, value))
    spinbox.setValue(value)
    editor.connect_value_signal(
        spinbox,
        spinbox.valueChanged,
        float,
        _subplots_adjust_kwarg_update_callback(editor, key),
    )
    return spinbox


def _method_control_visible(
    operation: FigureOperationState, spec: MethodSpec, control: MethodControlSpec
) -> bool:
    if not _is_layout_engine_method(spec) or control.key is None:
        return True
    return control.key in _layout_engine_kwarg_keys(
        _layout_engine_name(operation, spec)
    )


def _control_arg_index(control: MethodControlSpec) -> int:
    if control.arg_index is None:
        raise ValueError(f"{control.label} has no argument index")
    return control.arg_index


def _control_key(control: MethodControlSpec) -> str:
    if control.key is None:
        raise ValueError(f"{control.label} has no keyword name")
    return control.key


def _method_bool_arg_callback(
    editor: FigureOperationEditor, index: int
) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_arg(editor, index, text == "True")

    return update


def _method_arg_callback(
    editor: FigureOperationEditor, index: int, spec: MethodSpec
) -> Callable[[str], None]:
    def update(text: str) -> None:
        if _is_layout_engine_method(spec):
            _update_current_layout_engine(editor, index, text)
            return
        _update_current_method_arg(editor, index, text)

    return update


def _method_bool_kwarg_callback(
    editor: FigureOperationEditor, key: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_kwarg(editor, key, text == "True")

    return update


def _method_optional_bool_kwarg_callback(
    editor: FigureOperationEditor, key: str
) -> Callable[[str | None], None]:
    def update(text: str | None) -> None:
        value = None if text is None else text == "True"
        _update_current_method_kwarg(editor, key, value)

    return update


def _method_kwarg_callback(
    editor: FigureOperationEditor, key: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_kwarg(editor, key, text)

    return update


def _method_optional_kwarg_callback(
    editor: FigureOperationEditor, key: str
) -> Callable[[str | None], None]:
    def update(text: str | None) -> None:
        _update_current_method_kwarg(editor, key, text)

    return update


def _update_current_layout_engine(
    editor: FigureOperationEditor, index: int, text: str
) -> None:
    allowed_keys = _layout_engine_kwarg_keys(text)

    def update_engine(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        args = list(_method_args(operation, _method_spec(operation)))
        while len(args) <= index:
            args.append(None)
        args[index] = None if text == "default" else text
        kwargs = {
            key: value
            for key, value in operation.method_kwargs.items()
            if key in allowed_keys
        }
        return operation.model_copy(
            update={"method_args": tuple(args), "method_kwargs": kwargs}
        )

    editor.request_transform(
        update_engine,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _family_from_label(text: str) -> FigureMethodFamily:
    for family, label in _FAMILY_LABELS.items():
        if label == text:
            return family
    return FigureMethodFamily.ERLAB


def _method_combo_object_name(family: FigureMethodFamily) -> str:
    match family:
        case FigureMethodFamily.AXES:
            return "figureComposerAxesMethodCombo"
        case FigureMethodFamily.FIGURE:
            return "figureComposerFigureMethodCombo"
        case FigureMethodFamily.ERLAB:
            return "figureComposerERLabMethodCombo"


def _method_kwargs_object_name(family: FigureMethodFamily) -> str:
    match family:
        case FigureMethodFamily.AXES:
            return "figureComposerAxesMethodKwEdit"
        case FigureMethodFamily.FIGURE:
            return "figureComposerFigureMethodKwEdit"
        case FigureMethodFamily.ERLAB:
            return "figureComposerERLabMethodKwEdit"


def _open_method_doc_url(url: str) -> None:
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))


def _update_current_method_family(
    editor: FigureOperationEditor, family: FigureMethodFamily
) -> None:
    current = editor.current_operation()
    if current is not None and current[1].method_family == family:
        return
    spec = next(iter(_method_specs(family).values()))
    axes = (
        editor.selected_axes_state()
        if spec.target_domain == MethodTargetDomain.AXES
        else FigureAxesSelectionState(axes=())
    )
    editor.request_update_rebuild(
        label=spec.label,
        method_family=family,
        method_name=spec.name,
        method_args=_default_method_args(spec, editor.first_live_axis(axes)),
        method_kwargs={},
        method_call_policy=None,
        text_values=(),
        method_transform="data",
        method_transform_x="data",
        method_transform_y="axes",
        method_transform_expression="",
        axes=axes,
    )


def _update_current_method_name(editor: FigureOperationEditor, name: str) -> None:
    current = editor.current_operation()
    if current is None:
        return
    _index, operation = current
    if operation.method_name == name:
        return

    def update_method(
        _operation_index: int, target: FigureOperationState
    ) -> FigureOperationState:
        target_spec = _method_specs(target.method_family)[name]
        target_axes = (
            target.axes
            if target_spec.target_domain == MethodTargetDomain.AXES
            else FigureAxesSelectionState(axes=())
        )
        return target.model_copy(
            update=_method_transfer_updates(
                target,
                target_spec,
                default_axis=editor.first_live_axis(target_axes),
            )
        )

    editor.request_transform(
        update_method,
        rebuild_editor=True,
    )


def _update_current_method_args(
    editor: FigureOperationEditor, args: Sequence[typing.Any]
) -> None:
    editor.request_update(method_args=tuple(args))


def _update_current_method_arg(
    editor: FigureOperationEditor, index: int, value: typing.Any
) -> None:
    def update_arg(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        args = list(_method_args(operation, _method_spec(operation)))
        while len(args) <= index:
            args.append(None)
        args[index] = value
        return operation.model_copy(update={"method_args": tuple(args)})

    editor.request_transform(update_arg)


def _update_current_method_string_tuple_arg(
    editor: FigureOperationEditor, index: int, text: str
) -> None:
    values = _string_tuple_from_text(text)

    def update_arg(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        args = list(_method_args(operation, _method_spec(operation)))
        if values:
            while len(args) <= index:
                args.append(())
            args[index] = values
        elif len(args) > index:
            args = args[:index]
        return operation.model_copy(update={"method_args": tuple(args)})

    editor.request_transform(update_arg)


def _update_current_method_kwarg(
    editor: FigureOperationEditor, key: str, value: typing.Any
) -> None:
    def update_kwarg(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        kwargs = dict(operation.method_kwargs)
        if value is None:
            kwargs.pop(key, None)
        else:
            kwargs[key] = value
        return operation.model_copy(update={"method_kwargs": kwargs})

    editor.request_transform(update_kwarg)


def _update_current_subplots_adjust_kwarg(
    editor: FigureOperationEditor, key: str, value: typing.Any
) -> None:
    defaults = {
        name: editor.subplot_parameter_default(name)
        for name in ("left", "bottom", "right", "top", "wspace", "hspace")
    }

    def update_kwarg(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        values = _subplots_adjust_values(operation, defaults)
        kwargs = dict(operation.method_kwargs)
        if value is None:
            kwargs.pop(key, None)
        else:
            kwargs[key] = value
        kwargs = normalize_subplots_adjust_kwargs(
            kwargs,
            defaults=values,
            changed_key=key,
        )
        return operation.model_copy(update={"method_kwargs": kwargs})

    editor.request_transform(update_kwarg)


def _call_policy_from_label(text: str) -> MethodCallPolicy:
    for policy, label in _CALL_POLICY_LABELS.items():
        if label == text:
            return policy
    return MethodCallPolicy(text)


def _update_current_method_call_policy(
    editor: FigureOperationEditor, policy: MethodCallPolicy
) -> None:
    def update_policy(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        spec = _method_spec(operation)
        return operation.model_copy(
            update={
                "method_call_policy": None
                if policy == spec.call_policy
                else policy.value
            }
        )

    editor.request_transform(update_policy)


def _update_current_method_text_values(
    editor: FigureOperationEditor, text: str
) -> None:
    def update_text_values(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        spec = _method_spec(operation)
        return operation.model_copy(
            update={
                "text_values": _text_tuple_from_text(
                    text, preserve_empty=spec.preserves_empty_text
                )
            }
        )

    editor.request_transform(update_text_values)
