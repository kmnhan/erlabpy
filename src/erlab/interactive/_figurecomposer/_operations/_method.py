"""Generic curated function/method call operation for Figure Composer.

This module owns the "method" recipe step: direct Matplotlib ``ax.*`` calls,
direct Matplotlib ``fig.*`` calls, and curated ``erlab.plotting`` helpers. New
method support should usually be added by editing the spec tables in this file,
not by adding operation-specific branches elsewhere in Figure Composer.

How to add a method
-------------------
1. Pick the family table:

   - ``AXES_METHODS`` for bound Matplotlib calls such as ``ax.text(...)``.
   - ``FIGURE_METHODS`` for bound Matplotlib calls such as ``fig.supxlabel(...)``.
   - ``ERLAB_METHODS`` for functions available from ``erlab.plotting``.

2. Choose the target domain:

   - ``AXES`` shows the axes selector and invalidates when selected axes disappear.
   - ``FIGURE`` hides the axes selector and targets the whole Matplotlib figure.
   - ``NONE`` is for future no-target helpers.

3. Choose the call policy. The policy determines both render behavior and generated
   code, so avoid duplicating this logic in UI code.

   - ``BOUND_EACH_AXIS``: ``for ax in axes: ax.method(...)``.
   - ``AXES_POSITIONAL``: ``eplt.helper(axes, ...)``.
   - ``AX_KEYWORD``: ``eplt.helper(..., ax=axes)``.
   - ``EACH_AXIS_AX_KEYWORD``: ``for ax in axes: eplt.helper(..., ax=ax)``.
   - ``BOUND_FIGURE``: ``fig.method(...)``.
   - ``FIG_KEYWORD``: ``eplt.helper(..., fig=fig)``.
   - ``PLAIN_CALL``: ``eplt.helper(...)``.

4. Add controls with the small builder helpers. ``*_arg`` controls write to
   ``method_args`` and ``*_kwarg`` controls write to ``method_kwargs``. Use keyword
   controls whenever a helper also receives ``ax`` or ``fig`` as an injected keyword;
   otherwise positional arguments can shift into the injected target parameter.

5. Set defaults deliberately:

   - ``default_args`` are fallback positional arguments used when a recipe has no
     stored ``method_args``.
   - ``default_kwargs`` are fallback keyword arguments used only while rendering and
     generating code. They should not be copied into saved recipe state unless the
     user edits the corresponding control.
   - Control ``default`` values are only the initial widget text/value.

6. If the method accepts a list of labels through the shared text editor, set
   ``text_values_policy``. ``POSITIONAL`` appends the labels as a positional
   argument; ``KWARG`` stores them under ``text_values_kwarg``.

Tests should exercise the public recipe behavior: select the method step, edit the
stable object-named widgets, assert ``method_args`` or ``method_kwargs``, and execute
or inspect generated code when the call form itself is product behavior.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import typing

import matplotlib.scale
from qtpy import QtCore, QtWidgets

import erlab.plotting as eplt
from erlab.interactive._figurecomposer import _rendering
from erlab.interactive._figurecomposer._code import _axes_code, _axes_sequence_code
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
    _empty_source_editor,
    _empty_source_names,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import (
    _code_args,
    _code_kwargs,
    _dict_from_text,
    _float_pair_from_text,
    _format_dict,
    _format_literal_sequence,
    _format_pair,
    _format_string_tuple,
    _literal_sequence_from_text,
    _RawCode,
    _string_tuple_from_text,
    _text_tuple_from_text,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


class MethodTargetDomain(enum.StrEnum):
    AXES = "axes"
    FIGURE = "figure"
    NONE = "none"


class MethodCallPolicy(enum.StrEnum):
    BOUND_EACH_AXIS = "bound_each_axis"
    AXES_POSITIONAL = "axes_positional"
    AX_KEYWORD = "ax_keyword"
    EACH_AXIS_AX_KEYWORD = "each_axis_ax_keyword"
    BOUND_FIGURE = "bound_figure"
    FIG_KEYWORD = "fig_keyword"
    PLAIN_CALL = "plain_call"


class MethodTextValuesPolicy(enum.StrEnum):
    NONE = "none"
    POSITIONAL = "positional"
    KWARG = "kwarg"


class MethodControlKind(enum.StrEnum):
    ARG_COMBO = "arg_combo"
    INT_ARG = "int_arg"
    FLOAT_ARG = "float_arg"
    TEXT_ARG = "text_arg"
    LITERAL_ARG = "literal_arg"
    LITERAL_SEQUENCE_ARG = "literal_sequence_arg"
    STRING_TUPLE_ARG = "string_tuple_arg"
    FLOAT_PAIR_ARGS = "float_pair_args"
    BOOL_ARG_COMBO = "bool_arg_combo"
    KWARG_COMBO = "kwarg_combo"
    BOOL_KWARG_COMBO = "bool_kwarg_combo"
    INT_KWARG = "int_kwarg"
    FLOAT_KWARG = "float_kwarg"
    TEXT_KWARG = "text_kwarg"
    LITERAL_KWARG = "literal_kwarg"
    STRING_TUPLE_KWARG = "string_tuple_kwarg"
    FLOAT_PAIR_KWARG = "float_pair_kwarg"
    COORDINATE_SYSTEM = "coordinate_system"


@dataclasses.dataclass(frozen=True)
class MethodControlSpec:
    kind: MethodControlKind
    label: str
    tooltip: str
    object_name: str
    arg_index: int | None = None
    key: str | None = None
    options: tuple[str, ...] = ()
    default: typing.Any = None


@dataclasses.dataclass(frozen=True)
class MethodSpec:
    family: FigureMethodFamily
    name: str
    label: str
    tooltip: str
    target_domain: MethodTargetDomain
    call_policy: MethodCallPolicy
    allowed_call_policies: tuple[MethodCallPolicy, ...] = ()
    default_args: tuple[typing.Any, ...] = ()
    default_kwargs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    controls: tuple[MethodControlSpec, ...] = ()
    allow_extra_kwargs: bool = True
    text_values_policy: MethodTextValuesPolicy = MethodTextValuesPolicy.NONE
    text_values_kwarg: str = "values"
    preserves_empty_text: bool = False
    callable_name: str | None = None

    @property
    def call_name(self) -> str:
        return self.callable_name or self.name

    @property
    def selectable_call_policies(self) -> tuple[MethodCallPolicy, ...]:
        if self.allowed_call_policies:
            return self.allowed_call_policies
        return (self.call_policy,)


def _float_arg(
    label: str, index: int, object_name: str, tooltip: str
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.FLOAT_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
    )


def _int_arg(
    label: str, index: int, object_name: str, tooltip: str, *, default: int = 0
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.INT_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
    )


def _text_arg(
    label: str, index: int, object_name: str, tooltip: str
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.TEXT_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
    )


def _literal_arg(
    label: str,
    index: int,
    object_name: str,
    tooltip: str,
    *,
    default: typing.Any = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.LITERAL_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
    )


def _literal_sequence_arg(
    label: str, index: int, object_name: str, tooltip: str
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.LITERAL_SEQUENCE_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
    )


def _string_tuple_arg(
    label: str, index: int, object_name: str, tooltip: str
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.STRING_TUPLE_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
    )


def _float_pair_args(label: str, object_name: str, tooltip: str) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.FLOAT_PAIR_ARGS,
        label=label,
        object_name=object_name,
        tooltip=tooltip,
    )


def _bool_arg_combo(
    label: str,
    index: int,
    object_name: str,
    tooltip: str,
    *,
    default: bool = True,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.BOOL_ARG_COMBO,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
        options=("True", "False"),
        default=default,
    )


def _arg_combo(
    label: str,
    index: int,
    options: Sequence[str],
    default: str,
    object_name: str,
    tooltip: str,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.ARG_COMBO,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
        options=tuple(options),
        default=default,
    )


def _kwarg_combo(
    label: str,
    key: str,
    options: Sequence[str],
    default: str,
    object_name: str,
    tooltip: str,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.KWARG_COMBO,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        options=tuple(options),
        default=default,
    )


def _bool_kwarg_combo(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    default: bool,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.BOOL_KWARG_COMBO,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        options=("True", "False"),
        default=default,
    )


def _int_kwarg(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    default: int | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.INT_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
    )


def _float_kwarg(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    default: float | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.FLOAT_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
    )


def _text_kwarg(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    default: str | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.TEXT_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
    )


def _literal_kwarg(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    default: typing.Any = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.LITERAL_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
    )


def _string_tuple_kwarg(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.STRING_TUPLE_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
    )


def _float_pair_kwarg(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    default: tuple[float, float] | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.FLOAT_PAIR_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
    )


def _coordinate_system_control() -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.COORDINATE_SYSTEM,
        label="Coordinates",
        object_name="figureComposerAxesMethodCoordCombo",
        tooltip="Use data coordinates or normalized axes coordinates.",
        options=("data", "axes"),
        default="data",
    )


_AXIS_OPTIONS = ("x", "y", "z")
_SCALE_OPTIONS = tuple(matplotlib.scale.get_scale_names())
_DEFAULT_SCALE = "log" if "log" in _SCALE_OPTIONS else _SCALE_OPTIONS[0]
_FLATTEN_ORDER_OPTIONS = ("C", "F", "A", "K")
_COLORBAR_ORIENTATION_OPTIONS = ("vertical", "horizontal")
_LINE_ORIENTATION_OPTIONS = ("h", "v")
_LABEL_LOCATION_OPTIONS = (
    "upper left",
    "upper center",
    "upper right",
    "center left",
    "center",
    "center right",
    "lower left",
    "lower center",
    "lower right",
)
_FONT_WEIGHT_OPTIONS = (
    "ultralight",
    "light",
    "normal",
    "regular",
    "book",
    "medium",
    "roman",
    "semibold",
    "demibold",
    "demi",
    "bold",
    "heavy",
    "extra bold",
    "black",
)


AXES_METHODS: dict[str, MethodSpec] = {
    "text": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="text",
        label="Text",
        tooltip="Runs ax.text on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(0.5, 0.5, "Text"),
        controls=(
            _coordinate_system_control(),
            _float_arg(
                "x",
                0,
                "figureComposerAxesMethodXEdit",
                "x argument passed to ax.text.",
            ),
            _float_arg(
                "y",
                1,
                "figureComposerAxesMethodYEdit",
                "y argument passed to ax.text.",
            ),
            _text_arg(
                "Text",
                2,
                "figureComposerAxesMethodTextEdit",
                "Text string passed as the third ax.text argument.",
            ),
        ),
    ),
    "axvline": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="axvline",
        label="Vertical line",
        tooltip="Runs ax.axvline on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(0.0,),
        controls=(
            _float_arg(
                "x",
                0,
                "figureComposerAxesMethodXEdit",
                "x coordinate for the vertical line.",
            ),
        ),
    ),
    "axhline": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="axhline",
        label="Horizontal line",
        tooltip="Runs ax.axhline on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(0.0,),
        controls=(
            _float_arg(
                "y",
                0,
                "figureComposerAxesMethodYEdit",
                "y coordinate for the horizontal line.",
            ),
        ),
    ),
    "axvspan": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="axvspan",
        label="Vertical span",
        tooltip="Runs ax.axvspan on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(0.0, 1.0),
        controls=(
            _float_arg(
                "xmin",
                0,
                "figureComposerAxesMethodXminEdit",
                "Lower x boundary for the vertical span.",
            ),
            _float_arg(
                "xmax",
                1,
                "figureComposerAxesMethodXmaxEdit",
                "Upper x boundary for the vertical span.",
            ),
        ),
    ),
    "axhspan": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="axhspan",
        label="Horizontal span",
        tooltip="Runs ax.axhspan on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(0.0, 1.0),
        controls=(
            _float_arg(
                "ymin",
                0,
                "figureComposerAxesMethodYminEdit",
                "Lower y boundary for the horizontal span.",
            ),
            _float_arg(
                "ymax",
                1,
                "figureComposerAxesMethodYmaxEdit",
                "Upper y boundary for the horizontal span.",
            ),
        ),
    ),
    "set_xticks": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_xticks",
        label="Set x ticks",
        tooltip="Runs ax.set_xticks on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=((0.0, 1.0),),
        controls=(
            _literal_sequence_arg(
                "Tick values",
                0,
                "figureComposerAxesMethodTickValuesEdit",
                "Comma-separated tick positions or a Python list/tuple literal.",
            ),
            _string_tuple_arg(
                "Tick labels",
                1,
                "figureComposerAxesMethodTickLabelsEdit",
                "Optional comma-separated labels passed as set_ticks labels.",
            ),
        ),
    ),
    "set_yticks": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_yticks",
        label="Set y ticks",
        tooltip="Runs ax.set_yticks on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=((0.0, 1.0),),
        controls=(
            _literal_sequence_arg(
                "Tick values",
                0,
                "figureComposerAxesMethodTickValuesEdit",
                "Comma-separated tick positions or a Python list/tuple literal.",
            ),
            _string_tuple_arg(
                "Tick labels",
                1,
                "figureComposerAxesMethodTickLabelsEdit",
                "Optional comma-separated labels passed as set_ticks labels.",
            ),
        ),
    ),
    "set_xlim": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_xlim",
        label="Set x limits",
        tooltip="Runs ax.set_xlim on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(0.0, 1.0),
        controls=(
            _float_pair_args(
                "Limits",
                "figureComposerAxesMethodLimitsEdit",
                "Lower and upper limits as two comma-separated values.",
            ),
        ),
    ),
    "set_ylim": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_ylim",
        label="Set y limits",
        tooltip="Runs ax.set_ylim on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(0.0, 1.0),
        controls=(
            _float_pair_args(
                "Limits",
                "figureComposerAxesMethodLimitsEdit",
                "Lower and upper limits as two comma-separated values.",
            ),
        ),
    ),
    "set_xscale": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_xscale",
        label="Set x scale",
        tooltip="Runs ax.set_xscale on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(_DEFAULT_SCALE,),
        controls=(
            _arg_combo(
                "Scale",
                0,
                _SCALE_OPTIONS,
                _DEFAULT_SCALE,
                "figureComposerAxesMethodXScaleCombo",
                "Matplotlib scale name passed to ax.set_xscale.",
            ),
        ),
    ),
    "set_yscale": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_yscale",
        label="Set y scale",
        tooltip="Runs ax.set_yscale on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(_DEFAULT_SCALE,),
        controls=(
            _arg_combo(
                "Scale",
                0,
                _SCALE_OPTIONS,
                _DEFAULT_SCALE,
                "figureComposerAxesMethodYScaleCombo",
                "Matplotlib scale name passed to ax.set_yscale.",
            ),
        ),
    ),
    "grid": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="grid",
        label="Grid",
        tooltip="Runs ax.grid on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=(True,),
        controls=(
            _bool_arg_combo(
                "Visible",
                0,
                "figureComposerAxesMethodGridVisibleCombo",
                "Turn grid lines on or off.",
            ),
            _kwarg_combo(
                "which",
                "which",
                ("major", "minor", "both"),
                "major",
                "figureComposerAxesMethodWhichCombo",
                "Grid tick group.",
            ),
            _kwarg_combo(
                "axis",
                "axis",
                ("both", "x", "y"),
                "both",
                "figureComposerAxesMethodAxisCombo",
                "Grid axis direction.",
            ),
        ),
    ),
    "set_axis_off": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_axis_off",
        label="Hide axis",
        tooltip="Runs ax.set_axis_off on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        allow_extra_kwargs=False,
    ),
    "set_axis_on": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_axis_on",
        label="Show axis",
        tooltip="Runs ax.set_axis_on on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        allow_extra_kwargs=False,
    ),
    "invert_xaxis": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="invert_xaxis",
        label="Invert x axis",
        tooltip="Runs ax.invert_xaxis on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        allow_extra_kwargs=False,
    ),
    "invert_yaxis": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="invert_yaxis",
        label="Invert y axis",
        tooltip="Runs ax.invert_yaxis on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        allow_extra_kwargs=False,
    ),
}

FIGURE_METHODS: dict[str, MethodSpec] = {
    "supxlabel": MethodSpec(
        family=FigureMethodFamily.FIGURE,
        name="supxlabel",
        label="Figure x label",
        tooltip="Runs fig.supxlabel on the whole figure.",
        target_domain=MethodTargetDomain.FIGURE,
        call_policy=MethodCallPolicy.BOUND_FIGURE,
        default_args=("x label",),
        controls=(
            _text_arg(
                "Text",
                0,
                "figureComposerFigureMethodTextEdit",
                "Text string passed to fig.supxlabel.",
            ),
        ),
    ),
    "supylabel": MethodSpec(
        family=FigureMethodFamily.FIGURE,
        name="supylabel",
        label="Figure y label",
        tooltip="Runs fig.supylabel on the whole figure.",
        target_domain=MethodTargetDomain.FIGURE,
        call_policy=MethodCallPolicy.BOUND_FIGURE,
        default_args=("y label",),
        controls=(
            _text_arg(
                "Text",
                0,
                "figureComposerFigureMethodTextEdit",
                "Text string passed to fig.supylabel.",
            ),
        ),
    ),
    "suptitle": MethodSpec(
        family=FigureMethodFamily.FIGURE,
        name="suptitle",
        label="Figure title",
        tooltip="Runs fig.suptitle on the whole figure.",
        target_domain=MethodTargetDomain.FIGURE,
        call_policy=MethodCallPolicy.BOUND_FIGURE,
        default_args=("Title",),
        controls=(
            _text_arg(
                "Text",
                0,
                "figureComposerFigureMethodTextEdit",
                "Text string passed to fig.suptitle.",
            ),
        ),
    ),
}

ERLAB_METHODS: dict[str, MethodSpec] = {
    "clean_labels": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="clean_labels",
        label="clean_labels",
        tooltip="Runs erlab.plotting.clean_labels on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        controls=(
            _bool_arg_combo(
                "Remove inner ticks",
                0,
                "figureComposerERLabCleanLabelsRemoveInnerTicksCombo",
                "Also remove inner tick marks, not only inner tick labels.",
                default=False,
            ),
        ),
    ),
    "fancy_labels": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="fancy_labels",
        label="fancy_labels",
        tooltip="Runs erlab.plotting.fancy_labels on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        allow_extra_kwargs=False,
        controls=(
            _bool_kwarg_combo(
                "Radians",
                "radians",
                "figureComposerERLabFancyLabelsRadiansCombo",
                "Use radians instead of degrees for angular axis units.",
                default=False,
            ),
        ),
    ),
    "integer_ticks": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="integer_ticks",
        label="integer_ticks",
        tooltip="Runs erlab.plotting.integer_ticks on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        allow_extra_kwargs=False,
    ),
    "label_subplots": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="label_subplots",
        label="label_subplots",
        tooltip="Runs erlab.plotting.label_subplots on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        text_values_policy=MethodTextValuesPolicy.KWARG,
        controls=(
            _int_kwarg(
                "Start from",
                "startfrom",
                "figureComposerERLabLabelSubplotsStartEdit",
                "First number used for automatically generated labels.",
                default=1,
            ),
            _kwarg_combo(
                "Order",
                "order",
                _FLATTEN_ORDER_OPTIONS,
                "C",
                "figureComposerERLabLabelSubplotsOrderCombo",
                "Flattening order used to match labels to axes.",
            ),
            _kwarg_combo(
                "Location",
                "loc",
                _LABEL_LOCATION_OPTIONS,
                "upper left",
                "figureComposerERLabLabelSubplotsLocCombo",
                "Location of the anchored subplot label.",
            ),
            _float_pair_kwarg(
                "Offset",
                "offset",
                "figureComposerERLabLabelSubplotsOffsetEdit",
                "Label offset in display points as dx, dy.",
                default=(0.0, 0.0),
            ),
            _text_kwarg(
                "Prefix",
                "prefix",
                "figureComposerERLabLabelSubplotsPrefixEdit",
                "Text prepended to automatically generated labels.",
                default="",
            ),
            _text_kwarg(
                "Suffix",
                "suffix",
                "figureComposerERLabLabelSubplotsSuffixEdit",
                "Text appended to automatically generated labels.",
                default="",
            ),
            _bool_kwarg_combo(
                "Numeric labels",
                "numeric",
                "figureComposerERLabLabelSubplotsNumericCombo",
                "Use numbers instead of letters for generated labels.",
                default=False,
            ),
            _bool_kwarg_combo(
                "Capital letters",
                "capital",
                "figureComposerERLabLabelSubplotsCapitalCombo",
                "Use capital letters for generated alphabetic labels.",
                default=False,
            ),
            _kwarg_combo(
                "Font weight",
                "fontweight",
                _FONT_WEIGHT_OPTIONS,
                "normal",
                "figureComposerERLabLabelSubplotsFontWeightCombo",
                "Font weight for subplot labels.",
            ),
            _literal_kwarg(
                "Font size",
                "fontsize",
                "figureComposerERLabLabelSubplotsFontSizeEdit",
                "Matplotlib font size. Use 8 or quoted names such as 'large'.",
            ),
        ),
    ),
    "label_subplot_properties": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="label_subplot_properties",
        label="label_subplot_properties",
        tooltip="Runs erlab.plotting.label_subplot_properties on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        default_args=({"value": [0]},),
        controls=(
            _literal_arg(
                "Values",
                0,
                "figureComposerERLabLabelPropertiesValuesEdit",
                "Dictionary of property values, such as energy=[0, 1].",
                default={"value": [0]},
            ),
            _int_kwarg(
                "Decimals",
                "decimals",
                "figureComposerERLabLabelPropertiesDecimalsEdit",
                "Decimal places for formatted values. Blank leaves values unrounded.",
            ),
            _int_kwarg(
                "SI exponent",
                "si",
                "figureComposerERLabLabelPropertiesSiEdit",
                "Power of ten used for SI prefix formatting.",
                default=0,
            ),
            _text_kwarg(
                "Name",
                "name",
                "figureComposerERLabLabelPropertiesNameEdit",
                "Override the automatic property name in generated labels.",
            ),
            _text_kwarg(
                "Unit",
                "unit",
                "figureComposerERLabLabelPropertiesUnitEdit",
                "Override the automatic unit in generated labels.",
            ),
            _kwarg_combo(
                "Order",
                "order",
                _FLATTEN_ORDER_OPTIONS,
                "C",
                "figureComposerERLabLabelPropertiesOrderCombo",
                "Flattening order used to match property values to axes.",
            ),
        ),
    ),
    "nice_colorbar": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="nice_colorbar",
        label="nice_colorbar",
        tooltip="Runs erlab.plotting.nice_colorbar once for each selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        allowed_call_policies=(
            MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
            MethodCallPolicy.AX_KEYWORD,
        ),
        controls=(
            _float_kwarg(
                "Width",
                "width",
                "figureComposerERLabNiceColorbarWidthEdit",
                "Colorbar width in points.",
                default=8.0,
            ),
            _float_kwarg(
                "Aspect",
                "aspect",
                "figureComposerERLabNiceColorbarAspectEdit",
                "Colorbar aspect ratio.",
                default=5.0,
            ),
            _float_kwarg(
                "Pad",
                "pad",
                "figureComposerERLabNiceColorbarPadEdit",
                "Padding between axes and colorbar in points.",
                default=3.0,
            ),
            _bool_kwarg_combo(
                "Min/max ticks",
                "minmax",
                "figureComposerERLabNiceColorbarMinMaxCombo",
                "Label the minimum and maximum of the colorbar.",
                default=False,
            ),
            _kwarg_combo(
                "Orientation",
                "orientation",
                _COLORBAR_ORIENTATION_OPTIONS,
                "vertical",
                "figureComposerERLabNiceColorbarOrientationCombo",
                "Colorbar orientation.",
            ),
            _bool_kwarg_combo(
                "Floating inset",
                "floating",
                "figureComposerERLabNiceColorbarFloatingCombo",
                "Draw the colorbar as an inset anchored to the axes.",
                default=False,
            ),
            _literal_kwarg(
                "Ticks",
                "ticks",
                "figureComposerERLabNiceColorbarTicksEdit",
                "Optional colorbar tick locations.",
            ),
            _string_tuple_kwarg(
                "Tick labels",
                "ticklabels",
                "figureComposerERLabNiceColorbarTickLabelsEdit",
                "Optional comma-separated colorbar tick labels.",
            ),
        ),
    ),
    "proportional_colorbar": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="proportional_colorbar",
        label="proportional_colorbar",
        tooltip=(
            "Runs erlab.plotting.proportional_colorbar once for each selected axis."
        ),
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        allowed_call_policies=(
            MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
            MethodCallPolicy.AX_KEYWORD,
        ),
        controls=(
            _int_kwarg(
                "Mappable index",
                "index",
                "figureComposerERLabProportionalColorbarIndexEdit",
                "Mappable index to use when inferring from the target axes.",
                default=-1,
            ),
            _bool_kwarg_combo(
                "Images only",
                "image_only",
                "figureComposerERLabProportionalColorbarImageOnlyCombo",
                "Only consider image mappables when inferring the colorbar target.",
                default=False,
            ),
            _literal_kwarg(
                "Ticks",
                "ticks",
                "figureComposerERLabProportionalColorbarTicksEdit",
                "Optional colorbar tick locations passed upstream.",
            ),
        ),
    ),
    "unify_clim": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="unify_clim",
        label="unify_clim",
        tooltip="Runs erlab.plotting.unify_clim on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        allow_extra_kwargs=False,
        controls=(
            _bool_kwarg_combo(
                "Images only",
                "image_only",
                "figureComposerERLabUnifyClimImageOnlyCombo",
                "Only consider image mappables when reading and applying limits.",
                default=False,
            ),
            _bool_kwarg_combo(
                "Autoscale data",
                "autoscale",
                "figureComposerERLabUnifyClimAutoscaleCombo",
                "Autoscale each mappable before determining shared color limits.",
                default=False,
            ),
            _float_kwarg(
                "Minimum",
                "vmin",
                "figureComposerERLabUnifyClimVminEdit",
                "Explicit lower color limit. Blank infers it from selected axes.",
            ),
            _float_kwarg(
                "Maximum",
                "vmax",
                "figureComposerERLabUnifyClimVmaxEdit",
                "Explicit upper color limit. Blank infers it from selected axes.",
            ),
        ),
    ),
    "set_titles": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="set_titles",
        label="set_titles",
        tooltip="Runs erlab.plotting.set_titles on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        text_values_policy=MethodTextValuesPolicy.POSITIONAL,
        preserves_empty_text=True,
        controls=(
            _kwarg_combo(
                "Order",
                "order",
                _FLATTEN_ORDER_OPTIONS,
                "C",
                "figureComposerERLabSetTitlesOrderCombo",
                "Flattening order used to match titles to axes.",
            ),
        ),
    ),
    "set_xlabels": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="set_xlabels",
        label="set_xlabels",
        tooltip="Runs erlab.plotting.set_xlabels on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        text_values_policy=MethodTextValuesPolicy.POSITIONAL,
        preserves_empty_text=True,
        controls=(
            _kwarg_combo(
                "Order",
                "order",
                _FLATTEN_ORDER_OPTIONS,
                "C",
                "figureComposerERLabSetXLabelsOrderCombo",
                "Flattening order used to match x labels to axes.",
            ),
        ),
    ),
    "set_ylabels": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="set_ylabels",
        label="set_ylabels",
        tooltip="Runs erlab.plotting.set_ylabels on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        text_values_policy=MethodTextValuesPolicy.POSITIONAL,
        preserves_empty_text=True,
        controls=(
            _kwarg_combo(
                "Order",
                "order",
                _FLATTEN_ORDER_OPTIONS,
                "C",
                "figureComposerERLabSetYLabelsOrderCombo",
                "Flattening order used to match y labels to axes.",
            ),
        ),
    ),
    "fermiline": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="fermiline",
        label="fermiline",
        tooltip="Runs erlab.plotting.fermiline once for each selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        controls=(
            _float_kwarg(
                "Value",
                "value",
                "figureComposerERLabFermilineValueEdit",
                "Coordinate where the line is drawn.",
                default=0.0,
            ),
            _kwarg_combo(
                "Orientation",
                "orientation",
                _LINE_ORIENTATION_OPTIONS,
                "h",
                "figureComposerERLabFermilineOrientationCombo",
                "Draw a horizontal or vertical reference line.",
            ),
        ),
    ),
    "mark_points": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="mark_points",
        label="mark_points",
        tooltip="Runs erlab.plotting.mark_points once for each selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        default_args=((0.0,), ("G",)),
        controls=(
            _literal_sequence_arg(
                "Points",
                0,
                "figureComposerERLabMarkPointsPointsEdit",
                "Comma-separated x positions or a Python list/tuple literal.",
            ),
            _string_tuple_arg(
                "Labels",
                1,
                "figureComposerERLabMarkPointsLabelsEdit",
                "Comma-separated labels matching the point positions.",
            ),
            _literal_kwarg(
                "Y",
                "y",
                "figureComposerERLabMarkPointsYEdit",
                "Label y position, or a sequence matching the points.",
                default=0.0,
            ),
            _float_pair_kwarg(
                "Pad",
                "pad",
                "figureComposerERLabMarkPointsPadEdit",
                "Text offset in points as dx, dy.",
                default=(0.0, 1.75),
            ),
            _bool_kwarg_combo(
                "Literal labels",
                "literal",
                "figureComposerERLabMarkPointsLiteralCombo",
                "Use labels exactly as typed instead of parsing point-label markup.",
                default=False,
            ),
            _bool_kwarg_combo(
                "Roman text",
                "roman",
                "figureComposerERLabMarkPointsRomanCombo",
                "Use roman text for parsed labels.",
                default=True,
            ),
            _bool_kwarg_combo(
                "Bar labels",
                "bar",
                "figureComposerERLabMarkPointsBarCombo",
                "Draw a bar over parsed labels.",
                default=False,
            ),
        ),
    ),
    "sizebar": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="sizebar",
        label="sizebar",
        tooltip="Runs erlab.plotting.sizebar once for each selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.EACH_AXIS_AX_KEYWORD,
        default_kwargs={"value": 1.0, "unit": "m"},
        controls=(
            _float_kwarg(
                "Value",
                "value",
                "figureComposerERLabSizebarValueEdit",
                "Physical length represented by the size bar.",
                default=1.0,
            ),
            _text_kwarg(
                "Unit",
                "unit",
                "figureComposerERLabSizebarUnitEdit",
                "Base SI unit used for the size bar label.",
                default="m",
            ),
            _int_kwarg(
                "SI exponent",
                "si",
                "figureComposerERLabSizebarSiEdit",
                "Power of ten used to choose the displayed SI prefix.",
                default=0,
            ),
            _float_kwarg(
                "Resolution",
                "resolution",
                "figureComposerERLabSizebarResolutionEdit",
                "Scale of the current axes coordinates in base units.",
                default=1.0,
            ),
            _int_kwarg(
                "Decimals",
                "decimals",
                "figureComposerERLabSizebarDecimalsEdit",
                "Decimal places displayed in the generated label.",
                default=0,
            ),
            _text_kwarg(
                "Label",
                "label",
                "figureComposerERLabSizebarLabelEdit",
                "Override the automatically generated size bar label.",
            ),
            _kwarg_combo(
                "Location",
                "loc",
                _LABEL_LOCATION_OPTIONS,
                "lower right",
                "figureComposerERLabSizebarLocCombo",
                "Location of the anchored size bar.",
            ),
            _float_kwarg(
                "Pad",
                "pad",
                "figureComposerERLabSizebarPadEdit",
                "Padding around the label and bar in font-size units.",
                default=0.1,
            ),
            _float_kwarg(
                "Border pad",
                "borderpad",
                "figureComposerERLabSizebarBorderPadEdit",
                "Padding between the size bar and its anchor box.",
                default=0.5,
            ),
            _float_kwarg(
                "Separation",
                "sep",
                "figureComposerERLabSizebarSepEdit",
                "Separation between the bar and label in points.",
                default=3.0,
            ),
            _bool_kwarg_combo(
                "Frame",
                "frameon",
                "figureComposerERLabSizebarFrameCombo",
                "Draw a frame around the size bar.",
                default=False,
            ),
        ),
    ),
    "scale_units": MethodSpec(
        family=FigureMethodFamily.ERLAB,
        name="scale_units",
        label="scale_units",
        tooltip="Runs erlab.plotting.scale_units on the selected axes.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.AXES_POSITIONAL,
        default_args=("x", 0),
        controls=(
            _arg_combo(
                "Axis",
                0,
                _AXIS_OPTIONS,
                "x",
                "figureComposerERLabScaleUnitsAxisCombo",
                "Axis whose ticks and label should be rescaled.",
            ),
            _int_arg(
                "SI exponent",
                1,
                "figureComposerERLabScaleUnitsSiEdit",
                "Power of ten corresponding to the desired SI prefix.",
            ),
            _bool_kwarg_combo(
                "Update label prefix",
                "prefix",
                "figureComposerERLabScaleUnitsPrefixCombo",
                "Update the unit prefix in the axis label when possible.",
                default=True,
            ),
            _bool_kwarg_combo(
                "Use power notation",
                "power",
                "figureComposerERLabScaleUnitsPowerCombo",
                "Use scientific power notation instead of an SI prefix.",
                default=False,
            ),
        ),
    ),
}

_METHODS_BY_FAMILY = {
    FigureMethodFamily.AXES: AXES_METHODS,
    FigureMethodFamily.FIGURE: FIGURE_METHODS,
    FigureMethodFamily.ERLAB: ERLAB_METHODS,
}

_FAMILY_LABELS = {
    FigureMethodFamily.AXES: "Axes Method",
    FigureMethodFamily.FIGURE: "Figure Method",
    FigureMethodFamily.ERLAB: "ERLab Method",
}

_FAMILY_TOOLTIPS = {
    FigureMethodFamily.AXES: (
        "Add a direct Matplotlib ax.* annotation or axes-control step."
    ),
    FigureMethodFamily.FIGURE: "Add a direct Matplotlib fig.* figure-level step.",
    FigureMethodFamily.ERLAB: "Add an erlab.plotting helper step after plotting.",
}

_CALL_POLICY_LABELS = {
    MethodCallPolicy.EACH_AXIS_AX_KEYWORD: "Each selected axis",
    MethodCallPolicy.AX_KEYWORD: "Selected axes together",
}


def _method_specs(family: FigureMethodFamily) -> dict[str, MethodSpec]:
    return _METHODS_BY_FAMILY[family]


def _method_spec(operation: FigureOperationState) -> MethodSpec:
    methods = _method_specs(operation.method_family)
    if operation.method_name in methods:
        return methods[operation.method_name]
    return next(iter(methods.values()))


def _effective_call_policy(
    operation: FigureOperationState, spec: MethodSpec
) -> MethodCallPolicy:
    if operation.method_call_policy is None:
        return spec.call_policy
    try:
        policy = MethodCallPolicy(operation.method_call_policy)
    except ValueError:
        return spec.call_policy
    if policy in spec.selectable_call_policies:
        return policy
    return spec.call_policy


def _method_label(operation: FigureOperationState) -> str:
    return _method_spec(operation).label


def _method_operation(
    tool: FigureComposerTool, family: FigureMethodFamily
) -> FigureOperationState:
    spec = next(iter(_method_specs(family).values()))
    axes = (
        tool._selected_axes_state()
        if spec.target_domain == MethodTargetDomain.AXES
        else FigureAxesSelectionState(axes=())
    )
    return FigureOperationState.method(
        family=family,
        name=spec.name,
        label=spec.label,
        axes=axes,
        args=spec.default_args,
    )


def _method_add_action(family: FigureMethodFamily) -> AddStepActionSpec:
    def create_operation(tool: FigureComposerTool) -> FigureOperationState:
        return _method_operation(tool, family)

    return AddStepActionSpec(
        action_id=f"method:{family.value}",
        text=_FAMILY_LABELS[family],
        tooltip=_FAMILY_TOOLTIPS[family],
        create_operation=create_operation,
    )


def _uses_axes(operation: FigureOperationState) -> bool:
    return _method_spec(operation).target_domain == MethodTargetDomain.AXES


def _target_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    spec = _method_spec(operation)
    if spec.target_domain == MethodTargetDomain.FIGURE:
        return "Figure"
    if spec.target_domain == MethodTargetDomain.NONE:
        return "none"
    return tool._axes_target_text(operation.axes)


def _has_invalid_target(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    return (
        _uses_axes(operation)
        and bool(operation.axes.invalid_axes(tool._recipe.setup))
        and not operation.axes.expression
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    prefix = "Needs axes: " if _has_invalid_target(tool, operation) else ""
    return f"{prefix}{_method_display(operation)}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    spec = _method_spec(operation)
    return f"{spec.tooltip}\nTargets: {_target_text(tool, operation)}"


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    if key == "axes":
        return _target_text(tool, operation)
    if key == "method":
        return ""
    return ""


def _build_method_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> Sequence[StepSection]:
    page, layout = tool._new_step_form_page("figureComposerMethodPage")
    tool.operation_editor = page
    tool.operation_editor_layout = layout
    spec = _method_spec(operation)

    family_combo = tool._combo(
        [label for _family, label in _FAMILY_LABELS.items()],
        _FAMILY_LABELS[operation.method_family],
        lambda text: _update_current_method_family(tool, _family_from_label(text)),
        parent=page,
    )
    family_combo.setObjectName("figureComposerMethodFamilyCombo")
    tool._add_form_row(
        layout,
        "Family",
        family_combo,
        (
            "Choose whether this step calls an ERLab helper, "
            "an Axes method, or a Figure method."
        ),
    )

    method_combo = tool._combo(
        [method.label for method in _method_specs(operation.method_family).values()],
        spec.label,
        lambda text: _update_current_method_name(
            tool, _method_name_from_label(operation.method_family, text)
        ),
        parent=page,
    )
    method_combo.setObjectName(_method_combo_object_name(operation.method_family))
    tool._add_form_row(
        layout,
        "Method",
        method_combo,
        "Function or method called by this recipe step.",
    )

    if len(spec.selectable_call_policies) > 1:
        policy = _effective_call_policy(operation, spec)
        policy_combo = tool._combo(
            [
                _CALL_POLICY_LABELS.get(item, item.value)
                for item in spec.selectable_call_policies
            ],
            _CALL_POLICY_LABELS.get(policy, policy.value),
            lambda text: _update_current_method_call_policy(
                tool, _call_policy_from_label(text)
            ),
            parent=page,
        )
        policy_combo.setObjectName("figureComposerMethodCallPolicyCombo")
        tool._add_form_row(
            layout,
            "Apply to",
            policy_combo,
            (
                "Choose whether this method receives all selected axes "
                "at once or runs once per axis."
            ),
        )

    if spec.text_values_policy != MethodTextValuesPolicy.NONE:
        text_edit = QtWidgets.QPlainTextEdit(page)
        text_edit.setPlainText("\n".join(operation.text_values))
        text_edit.setMaximumHeight(70)
        text_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_edit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_edit.setObjectName("figureComposerMethodTextValuesEdit")
        text_edit.textChanged.connect(
            lambda edit=text_edit: _update_current_method_text_values(
                tool, edit.toPlainText()
            )
        )
        tool._add_form_row(
            layout,
            "Text values",
            text_edit,
            "One text value per line for methods that apply labels or annotations.",
        )

    for control in spec.controls:
        _add_method_control_row(tool, layout, operation, spec, control)

    if not spec.controls and spec.text_values_policy == MethodTextValuesPolicy.NONE:
        label = QtWidgets.QLabel(f"{_callable_display(spec)} takes no values.", page)
        label.setWordWrap(True)
        tool._add_form_row(
            layout,
            "Action",
            label,
            "This method runs directly on its configured target.",
        )

    if spec.allow_extra_kwargs:
        kwargs_edit = tool._line_edit(
            _format_dict(operation.method_kwargs), parent=page
        )
        kwargs_edit.setObjectName(_method_kwargs_object_name(operation.method_family))
        kwargs_edit.editingFinished.connect(
            lambda edit=kwargs_edit: tool._update_current_operation(
                method_kwargs=_dict_from_text(edit.text())
            )
        )
        tool._add_form_row(
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
    tool: FigureComposerTool,
    layout: QtWidgets.QFormLayout,
    operation: FigureOperationState,
    spec: MethodSpec,
    control: MethodControlSpec,
) -> None:
    args = _method_args(operation, spec)
    match control.kind:
        case MethodControlKind.COORDINATE_SYSTEM:
            combo = tool._combo(
                control.options,
                operation.method_coordinate_system,
                lambda text: tool._update_current_operation(
                    method_coordinate_system=typing.cast(
                        'typing.Literal["data", "axes"]', text
                    )
                ),
                parent=layout.parentWidget(),
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.ARG_COMBO:
            index = _control_arg_index(control)
            combo = tool._combo(
                control.options,
                str(args[index]) if index < len(args) else str(control.default),
                _method_arg_callback(tool, index),
                parent=layout.parentWidget(),
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.INT_ARG:
            index = _control_arg_index(control)
            value = args[index] if index < len(args) else control.default
            edit = tool._line_edit(
                _format_int_value(value), parent=layout.parentWidget()
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, index=index: _update_current_method_arg(
                    tool, index, int(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_ARG:
            index = _control_arg_index(control)
            value = args[index] if index < len(args) else 0.0
            edit = tool._line_edit(f"{float(value):g}", parent=layout.parentWidget())
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, index=index: _update_current_method_arg(
                    tool, index, float(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.TEXT_ARG:
            index = _control_arg_index(control)
            edit = tool._line_edit(
                str(args[index]) if index < len(args) else "",
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, index=index: _update_current_method_arg(
                    tool, index, edit.text()
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.LITERAL_ARG:
            index = _control_arg_index(control)
            value = args[index] if index < len(args) else control.default
            edit = tool._line_edit(
                _format_literal_value(value), parent=layout.parentWidget()
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, index=index: _update_current_method_arg(
                    tool, index, _literal_value_from_text(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.LITERAL_SEQUENCE_ARG:
            index = _control_arg_index(control)
            value = args[index] if index < len(args) else ()
            text = (
                _format_literal_sequence(typing.cast("Sequence[typing.Any]", value))
                if isinstance(value, (list, tuple))
                else repr(value)
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, index=index: _update_current_method_arg(
                    tool, index, _literal_sequence_from_text(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.STRING_TUPLE_ARG:
            index = _control_arg_index(control)
            value = args[index] if index < len(args) else ()
            edit = tool._line_edit(
                _format_string_tuple(typing.cast("Sequence[str]", value))
                if isinstance(value, (list, tuple))
                else "",
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, index=index: _update_current_method_string_tuple_arg(
                    tool, index, edit.text()
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_PAIR_ARGS:
            pair = (
                (float(args[0]), float(args[1]))
                if len(args) >= 2
                else typing.cast("tuple[float, float] | None", None)
            )
            edit = tool._line_edit(_format_pair(pair), parent=layout.parentWidget())
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit: _update_current_method_args(
                    tool, _float_pair_from_text(edit.text()) or ()
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.BOOL_ARG_COMBO:
            index = _control_arg_index(control)
            combo = tool._combo(
                control.options,
                str(bool(args[index])) if index < len(args) else str(control.default),
                _method_bool_arg_callback(tool, index),
                parent=layout.parentWidget(),
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.KWARG_COMBO:
            key = _control_key(control)
            combo = tool._combo(
                control.options,
                str(operation.method_kwargs.get(key, control.default)),
                _method_kwarg_callback(tool, key),
                parent=layout.parentWidget(),
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.BOOL_KWARG_COMBO:
            key = _control_key(control)
            combo = tool._combo(
                control.options,
                str(bool(operation.method_kwargs.get(key, control.default))),
                _method_bool_kwarg_callback(tool, key),
                parent=layout.parentWidget(),
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.INT_KWARG:
            key = _control_key(control)
            edit = tool._line_edit(
                _format_int_value(operation.method_kwargs.get(key, control.default)),
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: _update_current_method_kwarg(
                    tool, key, _optional_int_from_text(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_KWARG:
            key = _control_key(control)
            edit = tool._line_edit(
                _format_float_value(operation.method_kwargs.get(key, control.default)),
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: _update_current_method_kwarg(
                    tool, key, _optional_float_from_text(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.TEXT_KWARG:
            key = _control_key(control)
            edit = tool._line_edit(
                str(operation.method_kwargs.get(key, control.default) or ""),
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: _update_current_method_kwarg(
                    tool, key, edit.text() or None
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.LITERAL_KWARG:
            key = _control_key(control)
            edit = tool._line_edit(
                _format_literal_value(
                    operation.method_kwargs.get(key, control.default)
                ),
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: _update_current_method_kwarg(
                    tool, key, _optional_literal_from_text(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.STRING_TUPLE_KWARG:
            key = _control_key(control)
            value = operation.method_kwargs.get(key, ())
            edit = tool._line_edit(
                _format_string_tuple(typing.cast("Sequence[str]", value))
                if isinstance(value, (list, tuple))
                else "",
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: _update_current_method_kwarg(
                    tool, key, _string_tuple_from_text(edit.text()) or None
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_PAIR_KWARG:
            key = _control_key(control)
            edit = tool._line_edit(
                _format_pair(operation.method_kwargs.get(key, control.default)),
                parent=layout.parentWidget(),
            )
            edit.setObjectName(control.object_name)
            edit.editingFinished.connect(
                lambda edit=edit, key=key: _update_current_method_kwarg(
                    tool, key, _float_pair_from_text(edit.text())
                )
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)


def _control_arg_index(control: MethodControlSpec) -> int:
    if control.arg_index is None:
        raise ValueError(f"{control.label} has no argument index")
    return control.arg_index


def _control_key(control: MethodControlSpec) -> str:
    if control.key is None:
        raise ValueError(f"{control.label} has no keyword name")
    return control.key


def _method_bool_arg_callback(
    tool: FigureComposerTool, index: int
) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_arg(tool, index, text == "True")

    return update


def _method_arg_callback(tool: FigureComposerTool, index: int) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_arg(tool, index, text)

    return update


def _method_bool_kwarg_callback(
    tool: FigureComposerTool, key: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_kwarg(tool, key, text == "True")

    return update


def _method_kwarg_callback(tool: FigureComposerTool, key: str) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_kwarg(tool, key, text)

    return update


def _format_int_value(value: typing.Any) -> str:
    return "" if value is None else str(int(value))


def _format_float_value(value: typing.Any) -> str:
    return "" if value is None else f"{float(value):g}"


def _format_literal_value(value: typing.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _format_dict(value)
    if isinstance(value, (list, tuple)):
        return _format_literal_sequence(value)
    return _code_args((value,))


def _literal_value_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or "=" in stripped:
        return _dict_from_text(stripped)
    return ast.literal_eval(stripped)


def _optional_literal_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    return _literal_value_from_text(stripped)


def _optional_float_from_text(text: str) -> float | None:
    stripped = text.strip()
    return None if not stripped else float(stripped)


def _optional_int_from_text(text: str) -> int | None:
    stripped = text.strip()
    return None if not stripped else int(stripped)


def _family_from_label(text: str) -> FigureMethodFamily:
    for family, label in _FAMILY_LABELS.items():
        if label == text:
            return family
    return FigureMethodFamily.ERLAB


def _method_name_from_label(family: FigureMethodFamily, text: str) -> str:
    for name, spec in _method_specs(family).items():
        if spec.label == text:
            return name
    return next(iter(_method_specs(family)))


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


def _method_display(operation: FigureOperationState) -> str:
    spec = _method_spec(operation)
    if operation.method_family == FigureMethodFamily.AXES:
        return f"ax.{spec.name}"
    if operation.method_family == FigureMethodFamily.FIGURE:
        return f"fig.{spec.name}"
    return spec.name


def _callable_display(spec: MethodSpec) -> str:
    match spec.call_policy:
        case MethodCallPolicy.BOUND_EACH_AXIS:
            return f"ax.{spec.name}"
        case MethodCallPolicy.BOUND_FIGURE:
            return f"fig.{spec.name}"
        case (
            MethodCallPolicy.AXES_POSITIONAL
            | MethodCallPolicy.AX_KEYWORD
            | MethodCallPolicy.EACH_AXIS_AX_KEYWORD
            | MethodCallPolicy.FIG_KEYWORD
            | MethodCallPolicy.PLAIN_CALL
        ):
            return f"erlab.plotting.{spec.call_name}"


def _update_current_method_family(
    tool: FigureComposerTool, family: FigureMethodFamily
) -> None:
    spec = next(iter(_method_specs(family).values()))
    axes = (
        tool._selected_axes_state()
        if spec.target_domain == MethodTargetDomain.AXES
        else FigureAxesSelectionState(axes=())
    )
    tool._update_current_operation_rebuild(
        label=spec.label,
        method_family=family,
        method_name=spec.name,
        method_args=spec.default_args,
        method_kwargs={},
        method_call_policy=None,
        text_values=(),
        method_coordinate_system="data",
        axes=axes,
    )


def _update_current_method_name(tool: FigureComposerTool, name: str) -> None:
    current = tool._current_operation()
    if current is None:
        return
    _index, operation = current
    spec = _method_specs(operation.method_family)[name]
    axes = (
        operation.axes
        if spec.target_domain == MethodTargetDomain.AXES
        else FigureAxesSelectionState(axes=())
    )
    tool._update_current_operation_rebuild(
        label=spec.label,
        method_name=spec.name,
        method_args=spec.default_args,
        method_kwargs={},
        method_call_policy=None,
        text_values=(),
        method_coordinate_system="data",
        axes=axes,
    )


def _update_current_method_args(
    tool: FigureComposerTool, args: Sequence[typing.Any]
) -> None:
    tool._update_current_operation(method_args=tuple(args))


def _update_current_method_arg(
    tool: FigureComposerTool, index: int, value: typing.Any
) -> None:
    current = tool._current_operation()
    if current is None:
        return
    _row, operation = current
    args = list(_method_args(operation, _method_spec(operation)))
    while len(args) <= index:
        args.append(None)
    args[index] = value
    tool._update_current_operation(method_args=tuple(args))


def _update_current_method_string_tuple_arg(
    tool: FigureComposerTool, index: int, text: str
) -> None:
    current = tool._current_operation()
    if current is None:
        return
    _row, operation = current
    args = list(_method_args(operation, _method_spec(operation)))
    values = _string_tuple_from_text(text)
    if values:
        while len(args) <= index:
            args.append(())
        args[index] = values
    elif len(args) > index:
        args = args[:index]
    tool._update_current_operation(method_args=tuple(args))


def _update_current_method_kwarg(
    tool: FigureComposerTool, key: str, value: typing.Any
) -> None:
    current = tool._current_operation()
    if current is None:
        return
    _row, operation = current
    kwargs = dict(operation.method_kwargs)
    if value is None:
        kwargs.pop(key, None)
    else:
        kwargs[key] = value
    tool._update_current_operation(method_kwargs=kwargs)


def _call_policy_from_label(text: str) -> MethodCallPolicy:
    for policy, label in _CALL_POLICY_LABELS.items():
        if label == text:
            return policy
    return MethodCallPolicy(text)


def _update_current_method_call_policy(
    tool: FigureComposerTool, policy: MethodCallPolicy
) -> None:
    current = tool._current_operation()
    if current is None:
        return
    _row, operation = current
    spec = _method_spec(operation)
    tool._update_current_operation(
        method_call_policy=None if policy == spec.call_policy else policy.value
    )


def _update_current_method_text_values(tool: FigureComposerTool, text: str) -> None:
    current = tool._current_operation()
    if current is None:
        return
    _index, operation = current
    spec = _method_spec(operation)
    tool._update_current_operation(
        text_values=_text_tuple_from_text(
            text, preserve_empty=spec.preserves_empty_text
        )
    )


def _method_args(
    operation: FigureOperationState, spec: MethodSpec
) -> tuple[typing.Any, ...]:
    return operation.method_args or spec.default_args


def _label_values(operation: FigureOperationState) -> str | list[str]:
    values = list(operation.text_values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return values


def _render_args_kwargs(
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    axis: Axes | None = None,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    args = list(_method_args(operation, spec))
    kwargs = dict(spec.default_kwargs)
    kwargs.update(operation.method_kwargs)
    if spec.text_values_policy == MethodTextValuesPolicy.POSITIONAL:
        args.append(_label_values(operation))
    elif (
        spec.text_values_policy == MethodTextValuesPolicy.KWARG
        and operation.text_values
    ):
        kwargs.setdefault(spec.text_values_kwarg, list(operation.text_values))
    if (
        axis is not None
        and spec.family == FigureMethodFamily.AXES
        and spec.name == "text"
        and operation.method_coordinate_system == "axes"
    ):
        kwargs["transform"] = axis.transAxes
    return tuple(args), kwargs


def _code_args_kwargs(
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    axis_transform: str | None = None,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    args = list(_method_args(operation, spec))
    kwargs = dict(spec.default_kwargs)
    kwargs.update(operation.method_kwargs)
    if spec.text_values_policy == MethodTextValuesPolicy.POSITIONAL:
        args.append(_label_values(operation))
    elif (
        spec.text_values_policy == MethodTextValuesPolicy.KWARG
        and operation.text_values
    ):
        kwargs.setdefault(spec.text_values_kwarg, list(operation.text_values))
    if (
        axis_transform is not None
        and spec.family == FigureMethodFamily.AXES
        and spec.name == "text"
        and operation.method_coordinate_system == "axes"
    ):
        kwargs["transform"] = _RawCode(axis_transform)
    return tuple(args), kwargs


def _erlab_callable(spec: MethodSpec) -> Callable[..., typing.Any]:
    return typing.cast("Callable[..., typing.Any]", getattr(eplt, spec.call_name))


def _render_method(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    figure: Figure,
    axs: np.ndarray,
) -> None:
    spec = _method_spec(operation)
    call_policy = _effective_call_policy(operation, spec)
    axes = None
    if spec.target_domain == MethodTargetDomain.AXES:
        axes = _rendering._axes_from_selection(
            tool, operation.axes, axs, for_plot_slices=False
        )

    match call_policy:
        case MethodCallPolicy.BOUND_EACH_AXIS:
            if axes is None:
                return
            for axis in _rendering._iter_axes(axes):
                args, kwargs = _render_args_kwargs(operation, spec, axis=axis)
                getattr(axis, spec.call_name)(*args, **kwargs)
        case MethodCallPolicy.AXES_POSITIONAL:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(operation, spec)
            _erlab_callable(spec)(axes, *args, **kwargs)
        case MethodCallPolicy.AX_KEYWORD:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(operation, spec)
            _erlab_callable(spec)(*args, ax=axes, **kwargs)
        case MethodCallPolicy.EACH_AXIS_AX_KEYWORD:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(operation, spec)
            for axis in _rendering._iter_axes(axes):
                _erlab_callable(spec)(*args, ax=axis, **kwargs)
        case MethodCallPolicy.BOUND_FIGURE:
            args, kwargs = _render_args_kwargs(operation, spec)
            getattr(figure, spec.call_name)(*args, **kwargs)
        case MethodCallPolicy.FIG_KEYWORD:
            args, kwargs = _render_args_kwargs(operation, spec)
            _erlab_callable(spec)(*args, fig=figure, **kwargs)
        case MethodCallPolicy.PLAIN_CALL:
            args, kwargs = _render_args_kwargs(operation, spec)
            _erlab_callable(spec)(*args, **kwargs)


def _call_code(
    name: str, args: Sequence[typing.Any], kwargs: dict[str, typing.Any]
) -> str:
    args_text = _code_args(args)
    kwargs_text = _code_kwargs(kwargs)
    call_parts = [part for part in (args_text, kwargs_text) if part]
    return f"{name}({', '.join(call_parts)})"


def _method_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    spec = _method_spec(operation)
    call_policy = _effective_call_policy(operation, spec)
    match call_policy:
        case MethodCallPolicy.BOUND_EACH_AXIS:
            args, kwargs = _code_args_kwargs(
                operation, spec, axis_transform="ax.transAxes"
            )
            call = _call_code(f"ax.{spec.call_name}", args, kwargs)
            return [
                f"for ax in {_axes_sequence_code(tool, operation.axes)}:",
                f"    {call}",
            ]
        case MethodCallPolicy.AXES_POSITIONAL:
            axes_code = _axes_code(tool, operation.axes, for_plot_slices=False)
            args, kwargs = _code_args_kwargs(operation, spec)
            args_text = _code_args((_RawCode(axes_code), *args))
            kwargs_text = _code_kwargs(kwargs)
            parts = [part for part in (args_text, kwargs_text) if part]
            return [f"eplt.{spec.call_name}({', '.join(parts)})"]
        case MethodCallPolicy.AX_KEYWORD:
            axes_code = _axes_code(tool, operation.axes, for_plot_slices=False)
            args, kwargs = _code_args_kwargs(operation, spec)
            kwargs["ax"] = _RawCode(axes_code)
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]
        case MethodCallPolicy.EACH_AXIS_AX_KEYWORD:
            args, kwargs = _code_args_kwargs(operation, spec)
            kwargs["ax"] = _RawCode("ax")
            call = f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"
            return [
                f"for ax in {_axes_sequence_code(tool, operation.axes)}:",
                f"    {call}",
            ]
        case MethodCallPolicy.BOUND_FIGURE:
            args, kwargs = _code_args_kwargs(operation, spec)
            return [_call_code(f"fig.{spec.call_name}", args, kwargs)]
        case MethodCallPolicy.FIG_KEYWORD:
            args, kwargs = _code_args_kwargs(operation, spec)
            kwargs["fig"] = _RawCode("fig")
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]
        case MethodCallPolicy.PLAIN_CALL:
            args, kwargs = _code_args_kwargs(operation, spec)
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]


def _call_parts(args: Sequence[typing.Any], kwargs: dict[str, typing.Any]) -> str:
    return ", ".join(part for part in (_code_args(args), _code_kwargs(kwargs)) if part)


def _required_imports(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> Sequence[str]:
    spec = _method_spec(operation)
    if spec.family == FigureMethodFamily.ERLAB:
        return ("import erlab.plotting as eplt",)
    return ()


SPEC = OperationSpec(
    kind=FigureOperationKind.METHOD,
    add_actions=(
        _method_add_action(FigureMethodFamily.ERLAB),
        _method_add_action(FigureMethodFamily.AXES),
        _method_add_action(FigureMethodFamily.FIGURE),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=_target_text,
    has_invalid_target=_has_invalid_target,
    uses_axes=_uses_axes,
    uses_source_section=_uses_no_source_section,
    source_names=_empty_source_names,
    build_source_editor=_empty_source_editor,
    build_editor_sections=_build_method_editor,
    section_summary=_section_summary,
    render=_render_method,
    code_lines=_method_code,
    required_imports=_required_imports,
)
