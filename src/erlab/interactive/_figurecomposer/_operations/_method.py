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

7. Documentation links are derived from the method family and callable name. Set
   ``doc_name`` only when the documented name differs, and ``doc_url`` only for
   exceptional URLs that do not follow the family template.

Tests should exercise the public recipe behavior: select the method step, edit the
stable object-named widgets, assert ``method_args`` or ``method_kwargs``, and execute
or inspect generated code when the call form itself is product behavior.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import typing

import matplotlib
import matplotlib.scale
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive.utils
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._code import _axes_code, _axes_sequence_code
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._line_style import (
    LINE_MARKER_OPTIONS,
    LINE_STYLE_DEFAULT_LABEL,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    normalize_style_value,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
    _empty_source_editor,
    _empty_source_names,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
    _render_preview,
)
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._subplot_adjust import (
    SUBPLOTS_ADJUST_SPINBOX_DECIMALS,
    SUBPLOTS_ADJUST_SPINBOX_STEP,
    normalize_subplots_adjust_kwargs,
    subplots_adjust_spinbox_range,
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
    _literal_from_text,
    _literal_sequence_from_text,
    _RawCode,
    _string_tuple_from_text,
    _text_tuple_from_text,
)
from erlab.interactive._figurecomposer._widgets import _ColorLineEditWidget

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes

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
    PLOT_DATA_ARGS = "plot_data_args"
    STRING_TUPLE_ARG = "string_tuple_arg"
    FLOAT_PAIR_ARGS = "float_pair_args"
    ASPECT_ARG = "aspect_arg"
    BOOL_ARG_COMBO = "bool_arg_combo"
    KWARG_COMBO = "kwarg_combo"
    BOOL_KWARG_COMBO = "bool_kwarg_combo"
    INT_KWARG = "int_kwarg"
    FLOAT_KWARG = "float_kwarg"
    SUBPLOTS_ADJUST_KWARG = "subplots_adjust_kwarg"
    TEXT_KWARG = "text_kwarg"
    LITERAL_KWARG = "literal_kwarg"
    STRING_TUPLE_KWARG = "string_tuple_kwarg"
    FLOAT_PAIR_KWARG = "float_pair_kwarg"
    TRANSFORM = "transform"
    COLOR_KWARG = "color_kwarg"


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
    minimum: int | float | None = None
    maximum: int | float | None = None
    decimals: int | None = None
    step: int | float | None = None
    none_label: str | None = None


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
    doc_name: str | None = None
    doc_url: str | None = None

    @property
    def call_name(self) -> str:
        return self.callable_name or self.name

    @property
    def selectable_call_policies(self) -> tuple[MethodCallPolicy, ...]:
        if self.allowed_call_policies:
            return self.allowed_call_policies
        return (self.call_policy,)


_INT_SPINBOX_MINIMUM = -1_000_000
_INT_SPINBOX_MAXIMUM = 1_000_000
_FLOAT_SPINBOX_MINIMUM = -1_000_000_000.0
_FLOAT_SPINBOX_MAXIMUM = 1_000_000_000.0
_FLOAT_SPINBOX_DECIMALS = 6
_FLOAT_SPINBOX_STEP = 0.1


def _float_arg(
    label: str,
    index: int,
    object_name: str,
    tooltip: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    decimals: int | None = None,
    step: float | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.FLOAT_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
        minimum=minimum,
        maximum=maximum,
        decimals=decimals,
        step=step,
    )


def _int_arg(
    label: str,
    index: int,
    object_name: str,
    tooltip: str,
    *,
    default: int = 0,
    minimum: int | None = None,
    maximum: int | None = None,
    step: int | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.INT_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
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


def _plot_data_args() -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.PLOT_DATA_ARGS,
        label="Data",
        object_name="figureComposerAxesMethodPlotData",
        tooltip=(
            "Literal sequences passed to ax.plot.\n"
            "Leave x blank to call ax.plot(y, ...)."
        ),
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


def _aspect_arg(
    label: str,
    index: int,
    object_name: str,
    tooltip: str,
    *,
    default: str | float = "equal",
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.ASPECT_ARG,
        label=label,
        arg_index=index,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
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
    default: typing.Any,
    object_name: str,
    tooltip: str,
    *,
    none_label: str | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.KWARG_COMBO,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        options=tuple(options),
        default=default,
        none_label=none_label,
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
    minimum: int | None = None,
    maximum: int | None = None,
    step: int | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.INT_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
    )


def _float_kwarg(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    decimals: int | None = None,
    step: float | None = None,
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.FLOAT_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        default=default,
        minimum=minimum,
        maximum=maximum,
        decimals=decimals,
        step=step,
    )


def _subplots_adjust_kwarg(
    label: str, key: str, object_name: str, tooltip: str
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.SUBPLOTS_ADJUST_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
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


def _color_kwarg(
    label: str, key: str, object_name: str, tooltip: str
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.COLOR_KWARG,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
    )


def _transform_control() -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.TRANSFORM,
        label="Transform",
        object_name="figureComposerMethodTransformModeCombo",
        tooltip=(
            "Coordinate transform for this method.\n"
            "Use custom only for trusted local expressions using ax, fig, "
            "or mtransforms."
        ),
        options=(
            "data",
            "axes",
            "figure",
            "dpi",
            "xaxis",
            "yaxis",
            "blend",
            "custom",
        ),
        default="data",
    )


_AXIS_OPTIONS = ("x", "y", "z")
_TRANSFORM_COMPONENT_OPTIONS = ("data", "axes", "figure", "dpi")
_SCALE_OPTIONS = tuple(matplotlib.scale.get_scale_names())
_DEFAULT_SCALE = "log" if "log" in _SCALE_OPTIONS else _SCALE_OPTIONS[0]
_FLATTEN_ORDER_OPTIONS = ("C", "F", "A", "K")
_COLORBAR_ORIENTATION_OPTIONS = ("vertical", "horizontal")
_LINE_ORIENTATION_OPTIONS = ("h", "v")
_LAYOUT_ENGINE_OPTIONS = ("default", "none", "tight", "constrained", "compressed")
_LAYOUT_ENGINE_KWARGS = {
    "tight": frozenset(("pad", "h_pad", "w_pad", "rect")),
    "constrained": frozenset(("h_pad", "w_pad", "hspace", "wspace", "rect")),
    "compressed": frozenset(("h_pad", "w_pad", "hspace", "wspace", "rect")),
    "none": frozenset[str](),
    "default": frozenset[str](),
}
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
_LEGEND_LOC_OPTIONS = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
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


def _legend_controls(prefix: str) -> tuple[MethodControlSpec, ...]:
    return (
        _kwarg_combo(
            "Location",
            "loc",
            _LEGEND_LOC_OPTIONS,
            "best",
            f"figureComposer{prefix}MethodLegendLocCombo",
            "Legend location passed as loc.",
        ),
        _int_kwarg(
            "Columns",
            "ncols",
            f"figureComposer{prefix}MethodLegendColumnsEdit",
            "Number of legend columns passed as ncols.",
            default=1,
            minimum=1,
            maximum=999,
        ),
        _text_kwarg(
            "Title",
            "title",
            f"figureComposer{prefix}MethodLegendTitleEdit",
            "Optional legend title.",
        ),
        _bool_kwarg_combo(
            "Frame",
            "frameon",
            f"figureComposer{prefix}MethodLegendFrameCombo",
            "Show or hide the legend frame.",
            default=True,
        ),
        _text_kwarg(
            "Font size",
            "fontsize",
            f"figureComposer{prefix}MethodLegendFontSizeEdit",
            "Legend text size.\nUse a named size or a numeric value.",
        ),
        _text_kwarg(
            "Title size",
            "title_fontsize",
            f"figureComposer{prefix}MethodLegendTitleFontSizeEdit",
            "Legend title text size.\nUse a named size or a numeric value.",
        ),
        _float_kwarg(
            "Marker scale",
            "markerscale",
            f"figureComposer{prefix}MethodLegendMarkerScaleEdit",
            "Scale factor for marker size in the legend.",
            default=float(matplotlib.rcParams["legend.markerscale"]),
            minimum=0.0,
            maximum=100.0,
            step=0.1,
        ),
        _float_kwarg(
            "Label spacing",
            "labelspacing",
            f"figureComposer{prefix}MethodLegendLabelSpacingEdit",
            "Vertical spacing between legend labels.",
            default=float(matplotlib.rcParams["legend.labelspacing"]),
            minimum=0.0,
            maximum=100.0,
            step=0.1,
        ),
        _float_kwarg(
            "Handle length",
            "handlelength",
            f"figureComposer{prefix}MethodLegendHandleLengthEdit",
            "Length of the legend handle.",
            default=float(matplotlib.rcParams["legend.handlelength"]),
            minimum=0.0,
            maximum=100.0,
            step=0.1,
        ),
        _float_kwarg(
            "Handle text pad",
            "handletextpad",
            f"figureComposer{prefix}MethodLegendHandleTextPadEdit",
            "Spacing between the handle and legend text.",
            default=float(matplotlib.rcParams["legend.handletextpad"]),
            minimum=0.0,
            maximum=100.0,
            step=0.1,
        ),
        _float_kwarg(
            "Column spacing",
            "columnspacing",
            f"figureComposer{prefix}MethodLegendColumnSpacingEdit",
            "Horizontal spacing between legend columns.",
            default=float(matplotlib.rcParams["legend.columnspacing"]),
            minimum=0.0,
            maximum=100.0,
            step=0.1,
        ),
        _literal_kwarg(
            "Anchor",
            "bbox_to_anchor",
            f"figureComposer{prefix}MethodLegendAnchorEdit",
            "Optional bbox_to_anchor value.\nUse a tuple like 1.0, 1.0.",
        ),
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
            _transform_control(),
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
    "plot": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="plot",
        label="Plot",
        tooltip="Runs ax.plot on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=((0.0, 1.0),),
        controls=(
            _plot_data_args(),
            _color_kwarg(
                "Color",
                "color",
                "figureComposerAxesMethodPlotColorEdit",
                "Matplotlib color for the plotted line.",
            ),
            _kwarg_combo(
                "Line style",
                "linestyle",
                LINE_STYLE_OPTIONS,
                None,
                "figureComposerAxesMethodPlotLineStyleCombo",
                "Matplotlib line style for the plotted line.",
                none_label=LINE_STYLE_DEFAULT_LABEL,
            ),
            _float_kwarg(
                "Line width",
                "linewidth",
                "figureComposerAxesMethodPlotLineWidthSpin",
                "Line width for the plotted line.",
                default=float(matplotlib.rcParams["lines.linewidth"]),
                minimum=0.0,
                maximum=1_000_000.0,
                step=0.5,
            ),
            _kwarg_combo(
                "Marker",
                "marker",
                LINE_MARKER_OPTIONS,
                None,
                "figureComposerAxesMethodPlotMarkerCombo",
                "Matplotlib marker style for the plotted line.",
                none_label=LINE_STYLE_DEFAULT_LABEL,
            ),
            _float_kwarg(
                "Marker size",
                "markersize",
                "figureComposerAxesMethodPlotMarkerSizeSpin",
                "Marker size for the plotted line.",
                default=float(matplotlib.rcParams["lines.markersize"]),
                minimum=0.0,
                maximum=1_000_000.0,
                step=0.5,
            ),
            _color_kwarg(
                "Marker face",
                "markerfacecolor",
                "figureComposerAxesMethodPlotMarkerFaceColorEdit",
                "Matplotlib marker face color.",
            ),
            _color_kwarg(
                "Marker edge",
                "markeredgecolor",
                "figureComposerAxesMethodPlotMarkerEdgeColorEdit",
                "Matplotlib marker edge color.",
            ),
            _float_kwarg(
                "Alpha",
                "alpha",
                "figureComposerAxesMethodPlotAlphaSpin",
                "Line opacity between 0 and 1.",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            ),
            _text_kwarg(
                "Label",
                "label",
                "figureComposerAxesMethodPlotLabelEdit",
                "Legend label for this plotted line.",
            ),
            _float_kwarg(
                "Z order",
                "zorder",
                "figureComposerAxesMethodPlotZOrderSpin",
                "Drawing order for this line.",
                default=2.0,
                step=1.0,
            ),
            _transform_control(),
        ),
    ),
    "legend": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="legend",
        label="Legend",
        tooltip="Runs ax.legend on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        controls=_legend_controls("Axes"),
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
        controls=(
            _float_pair_args(
                "Limits",
                "figureComposerAxesMethodLimitsEdit",
                "Lower and upper limits as two comma-separated values.",
            ),
        ),
    ),
    "set_title": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_title",
        label="Set title",
        tooltip="Runs ax.set_title on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=("Title",),
        controls=(
            _text_arg(
                "Title",
                0,
                "figureComposerAxesMethodTitleEdit",
                "Title text passed to ax.set_title.",
            ),
            _kwarg_combo(
                "Location",
                "loc",
                ("center", "left", "right"),
                "center",
                "figureComposerAxesMethodTitleLocCombo",
                "Horizontal title location.",
            ),
            _float_kwarg(
                "Padding",
                "pad",
                "figureComposerAxesMethodTitlePadEdit",
                "Optional title padding in points.",
                default=float(matplotlib.rcParams["axes.titlepad"]),
                minimum=0.0,
                maximum=1000.0,
                step=0.5,
            ),
        ),
    ),
    "set_xlabel": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_xlabel",
        label="Set x label",
        tooltip="Runs ax.set_xlabel on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=("x",),
        controls=(
            _text_arg(
                "Label",
                0,
                "figureComposerAxesMethodXLabelEdit",
                "x-axis label text passed to ax.set_xlabel.",
            ),
            _kwarg_combo(
                "Location",
                "loc",
                ("center", "left", "right"),
                "center",
                "figureComposerAxesMethodXLabelLocCombo",
                "x-axis label location.",
            ),
            _float_kwarg(
                "Label pad",
                "labelpad",
                "figureComposerAxesMethodXLabelPadEdit",
                "Optional x-axis label padding in points.",
                default=float(matplotlib.rcParams["axes.labelpad"]),
                minimum=0.0,
                maximum=1000.0,
                step=0.5,
            ),
        ),
    ),
    "set_ylabel": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_ylabel",
        label="Set y label",
        tooltip="Runs ax.set_ylabel on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=("y",),
        controls=(
            _text_arg(
                "Label",
                0,
                "figureComposerAxesMethodYLabelEdit",
                "y-axis label text passed to ax.set_ylabel.",
            ),
            _kwarg_combo(
                "Location",
                "loc",
                ("center", "bottom", "top"),
                "center",
                "figureComposerAxesMethodYLabelLocCombo",
                "y-axis label location.",
            ),
            _float_kwarg(
                "Label pad",
                "labelpad",
                "figureComposerAxesMethodYLabelPadEdit",
                "Optional y-axis label padding in points.",
                default=float(matplotlib.rcParams["axes.labelpad"]),
                minimum=0.0,
                maximum=1000.0,
                step=0.5,
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
    "margins": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="margins",
        label="Margins",
        tooltip="Runs ax.margins on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_kwargs={"x": 0.05, "y": 0.05, "tight": True},
        controls=(
            _float_kwarg(
                "x",
                "x",
                "figureComposerAxesMethodXMarginEdit",
                "x-axis margin fraction passed to ax.margins.",
                default=0.05,
                minimum=-0.49,
                maximum=10.0,
                decimals=4,
                step=0.01,
            ),
            _float_kwarg(
                "y",
                "y",
                "figureComposerAxesMethodYMarginEdit",
                "y-axis margin fraction passed to ax.margins.",
                default=0.05,
                minimum=-0.49,
                maximum=10.0,
                decimals=4,
                step=0.01,
            ),
            _bool_kwarg_combo(
                "Tight",
                "tight",
                "figureComposerAxesMethodMarginsTightCombo",
                "Pass tight=True or False to ax.margins.",
                default=True,
            ),
        ),
    ),
    "set_aspect": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="set_aspect",
        label="Set aspect",
        tooltip="Runs ax.set_aspect on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=("equal",),
        controls=(
            _aspect_arg(
                "Aspect",
                0,
                "figureComposerAxesMethodAspectEdit",
                "Aspect passed to ax.set_aspect: auto, equal, or a number.",
            ),
            _bool_kwarg_combo(
                "Share",
                "share",
                "figureComposerAxesMethodAspectShareCombo",
                "Apply the aspect setting to shared axes.",
                default=False,
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
    "legend": MethodSpec(
        family=FigureMethodFamily.FIGURE,
        name="legend",
        label="Legend",
        tooltip="Runs fig.legend for a figure-level legend.",
        target_domain=MethodTargetDomain.FIGURE,
        call_policy=MethodCallPolicy.BOUND_FIGURE,
        controls=_legend_controls("Figure"),
    ),
    "subplots_adjust": MethodSpec(
        family=FigureMethodFamily.FIGURE,
        name="subplots_adjust",
        label="Adjust subplots",
        tooltip="Runs fig.subplots_adjust on the whole figure.",
        target_domain=MethodTargetDomain.FIGURE,
        call_policy=MethodCallPolicy.BOUND_FIGURE,
        controls=(
            _subplots_adjust_kwarg(
                "Left",
                "left",
                "figureComposerFigureSubplotsAdjustLeftEdit",
                "Left edge of the subplots as a fraction of figure width.",
            ),
            _subplots_adjust_kwarg(
                "Bottom",
                "bottom",
                "figureComposerFigureSubplotsAdjustBottomEdit",
                "Bottom edge of the subplots as a fraction of figure height.",
            ),
            _subplots_adjust_kwarg(
                "Right",
                "right",
                "figureComposerFigureSubplotsAdjustRightEdit",
                "Right edge of the subplots as a fraction of figure width.",
            ),
            _subplots_adjust_kwarg(
                "Top",
                "top",
                "figureComposerFigureSubplotsAdjustTopEdit",
                "Top edge of the subplots as a fraction of figure height.",
            ),
            _subplots_adjust_kwarg(
                "Width spacing",
                "wspace",
                "figureComposerFigureSubplotsAdjustWspaceEdit",
                "Horizontal space between subplot groups.",
            ),
            _subplots_adjust_kwarg(
                "Height spacing",
                "hspace",
                "figureComposerFigureSubplotsAdjustHspaceEdit",
                "Vertical space between subplot groups.",
            ),
        ),
    ),
    "set_layout_engine": MethodSpec(
        family=FigureMethodFamily.FIGURE,
        name="set_layout_engine",
        label="Set layout engine",
        tooltip="Runs fig.set_layout_engine on the whole figure.",
        target_domain=MethodTargetDomain.FIGURE,
        call_policy=MethodCallPolicy.BOUND_FIGURE,
        default_args=("none",),
        controls=(
            _arg_combo(
                "Engine",
                0,
                _LAYOUT_ENGINE_OPTIONS,
                "none",
                "figureComposerFigureLayoutEngineCombo",
                "Matplotlib layout engine name.",
            ),
            _float_kwarg(
                "Pad",
                "pad",
                "figureComposerFigureLayoutEnginePadEdit",
                "Padding for the tight layout engine.",
                default=1.08,
                minimum=0.0,
                maximum=100.0,
                decimals=4,
                step=0.01,
            ),
            _float_kwarg(
                "Height pad",
                "h_pad",
                "figureComposerFigureLayoutEngineHpadEdit",
                "Height padding for tight, constrained, or compressed layout.",
                default=0.04167,
                minimum=0.0,
                maximum=100.0,
                decimals=5,
                step=0.01,
            ),
            _float_kwarg(
                "Width pad",
                "w_pad",
                "figureComposerFigureLayoutEngineWpadEdit",
                "Width padding for tight, constrained, or compressed layout.",
                default=0.04167,
                minimum=0.0,
                maximum=100.0,
                decimals=5,
                step=0.01,
            ),
            _float_kwarg(
                "Height spacing",
                "hspace",
                "figureComposerFigureLayoutEngineHspaceEdit",
                "Height spacing for constrained or compressed layout.",
                default=0.02,
                minimum=0.0,
                maximum=100.0,
                decimals=5,
                step=0.01,
            ),
            _float_kwarg(
                "Width spacing",
                "wspace",
                "figureComposerFigureLayoutEngineWspaceEdit",
                "Width spacing for constrained or compressed layout.",
                default=0.02,
                minimum=0.0,
                maximum=100.0,
                decimals=5,
                step=0.01,
            ),
            _literal_kwarg(
                "Rect",
                "rect",
                "figureComposerFigureLayoutEngineRectEdit",
                "Layout rectangle as left, bottom, width, height.",
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
                minimum=0,
                maximum=9999,
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
                minimum=-24,
                maximum=24,
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
                minimum=0.0,
                maximum=1000.0,
                step=0.5,
            ),
            _float_kwarg(
                "Aspect",
                "aspect",
                "figureComposerERLabNiceColorbarAspectEdit",
                "Colorbar aspect ratio.",
                default=5.0,
                minimum=0.0,
                maximum=1000.0,
                step=0.5,
            ),
            _float_kwarg(
                "Pad",
                "pad",
                "figureComposerERLabNiceColorbarPadEdit",
                "Padding between axes and colorbar in points.",
                default=3.0,
                minimum=0.0,
                maximum=1000.0,
                step=0.5,
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
                minimum=-1,
                maximum=999,
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
            _color_kwarg(
                "Line color",
                "color",
                "figureComposerERLabFermilineColorEdit",
                (
                    "Optional Matplotlib color for the line.\n"
                    "Leave blank to let fermiline use the axes text color."
                ),
            ),
            _kwarg_combo(
                "Line style",
                "linestyle",
                LINE_STYLE_OPTIONS,
                None,
                "figureComposerERLabFermilineLineStyleCombo",
                (
                    "Optional Matplotlib line style for the line.\n"
                    "Leave unset to let fermiline use its default style."
                ),
                none_label=LINE_STYLE_DEFAULT_LABEL,
            ),
            _float_kwarg(
                "Line width",
                "linewidth",
                "figureComposerERLabFermilineLineWidthEdit",
                (
                    "Optional line width for the line.\n"
                    "Leave blank to let fermiline use its default width."
                ),
                minimum=0.0,
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
                minimum=0.0,
                maximum=1_000_000_000.0,
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
                minimum=-24,
                maximum=24,
            ),
            _float_kwarg(
                "Resolution",
                "resolution",
                "figureComposerERLabSizebarResolutionEdit",
                "Scale of the current axes coordinates in base units.",
                default=1.0,
                minimum=0.0,
                maximum=1_000_000_000.0,
            ),
            _int_kwarg(
                "Decimals",
                "decimals",
                "figureComposerERLabSizebarDecimalsEdit",
                "Decimal places displayed in the generated label.",
                default=0,
                minimum=0,
                maximum=12,
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
                minimum=0.0,
                maximum=1000.0,
                step=0.1,
            ),
            _float_kwarg(
                "Border pad",
                "borderpad",
                "figureComposerERLabSizebarBorderPadEdit",
                "Padding between the size bar and its anchor box.",
                default=0.5,
                minimum=0.0,
                maximum=1000.0,
                step=0.1,
            ),
            _float_kwarg(
                "Separation",
                "sep",
                "figureComposerERLabSizebarSepEdit",
                "Separation between the bar and label in points.",
                default=3.0,
                minimum=0.0,
                maximum=1000.0,
                step=0.5,
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
                minimum=-24,
                maximum=24,
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
_MATPLOTLIB_DOC_BASE = "https://matplotlib.org/stable/api/_as_gen"
_ERLAB_PLOTTING_DOC_BASE = (
    "https://erlabpy.readthedocs.io/en/stable/erlab.plotting.html"
)

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


def _method_selector_text(spec: MethodSpec) -> str:
    return spec.name


def _method_combo(
    tool: FigureComposerTool,
    family: FigureMethodFamily,
    current_name: str,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox(parent)
    tool._mark_editor_control(combo)
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
            _update_current_method_name(tool, method_name)

    tool._connect_editor_signal(combo, combo.activated, method_activated)
    return combo


def _live_layout_axes(
    tool: FigureComposerTool,
    *,
    render_if_missing: bool = False,
) -> np.ndarray | dict[str, Axes] | None:
    setup = tool._recipe.setup
    if setup.layout_mode == "gridspec":
        axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
        axes = tool.figure.axes[: len(axes_ids)]
        if len(axes) < len(axes_ids):
            if render_if_missing:
                _render_preview(tool, show_window=False)
                return _live_layout_axes(tool)
            return None
        return dict(zip(axes_ids, axes, strict=True))

    count = setup.nrows * setup.ncols
    axes = tool.figure.axes[:count]
    if len(axes) < count:
        if render_if_missing:
            _render_preview(tool, show_window=False)
            return _live_layout_axes(tool)
        return None
    return np.asarray(axes, dtype=object).reshape(setup.nrows, setup.ncols)


def _first_live_axis(
    tool: FigureComposerTool, selection: FigureAxesSelectionState
) -> Axes | None:
    layout_axes = _live_layout_axes(tool, render_if_missing=True)
    if layout_axes is None:
        return None
    if isinstance(layout_axes, dict) and not selection.axes_ids:
        selection = selection.model_copy(
            update={
                "axes_ids": _gridspec_valid_axes_ids(
                    tool._recipe.setup, _gridspec_all_axes_ids(tool._recipe.setup)
                )[:1]
            }
        )
    try:
        selected_axes = _axes_from_selection(
            tool, selection, layout_axes, for_plot_slices=False
        )
    except (IndexError, TypeError, ValueError):
        return None
    axes = _iter_axes(selected_axes)
    return axes[0] if axes else None


def _limit_method_default_args(
    tool: FigureComposerTool,
    spec: MethodSpec,
    axes: FigureAxesSelectionState,
) -> tuple[typing.Any, ...]:
    axis = _first_live_axis(tool, axes)
    if axis is None:
        return ()
    limits = axis.get_xlim() if spec.name == "set_xlim" else axis.get_ylim()
    return float(limits[0]), float(limits[1])


def _default_method_args(
    tool: FigureComposerTool,
    spec: MethodSpec,
    axes: FigureAxesSelectionState,
) -> tuple[typing.Any, ...]:
    if spec.family == FigureMethodFamily.AXES and spec.name in {"set_xlim", "set_ylim"}:
        return _limit_method_default_args(tool, spec, axes)
    return spec.default_args


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
        args=_default_method_args(tool, spec, axes),
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
    return _uses_axes(operation) and tool._axes_selection_has_invalid_target(
        operation.axes
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

    method_widget = QtWidgets.QWidget(page)
    method_layout = QtWidgets.QHBoxLayout(method_widget)
    method_layout.setContentsMargins(0, 0, 0, 0)
    method_layout.setSpacing(6)
    method_combo = _method_combo(
        tool, operation.method_family, spec.name, method_widget
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
        tool._connect_editor_signal(
            docs_button,
            docs_button.clicked,
            lambda _checked=False, doc_url=doc_url: _open_method_doc_url(doc_url),
        )
    method_layout.addWidget(docs_button)
    tool._add_form_row(
        layout,
        "Method",
        method_widget,
        "Function or method called by this recipe step.",
    )

    if len(spec.selectable_call_policies) > 1:
        policy = _effective_call_policy(operation, spec)
        policy_mixed = tool._batch_is_mixed(
            operation, lambda target: _effective_call_policy(target, spec)
        )
        policy_combo = tool._combo(
            [
                _CALL_POLICY_LABELS.get(item, item.value)
                for item in spec.selectable_call_policies
            ],
            None if policy_mixed else _CALL_POLICY_LABELS.get(policy, policy.value),
            lambda text: _update_current_method_call_policy(
                tool, _call_policy_from_label(text)
            ),
            parent=page,
            mixed=policy_mixed,
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
        text_values_text, text_values_mixed = tool._batch_text(
            operation,
            lambda target: target.text_values,
            lambda value: "\n".join(typing.cast("Sequence[str]", value)),
        )
        text_edit = QtWidgets.QPlainTextEdit(page)
        text_edit.setPlainText(text_values_text)
        tool._apply_mixed_plain_text_edit(text_edit, text_values_mixed)
        text_edit.setMaximumHeight(70)
        text_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_edit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        text_edit.setObjectName("figureComposerMethodTextValuesEdit")
        tool._connect_plain_text_changed(
            text_edit,
            lambda text: _update_current_method_text_values(tool, text),
        )
        tool._add_form_row(
            layout,
            "Text values",
            text_edit,
            "One text value per line for methods that apply labels or annotations.",
        )

    for control in spec.controls:
        if _method_control_visible(operation, spec, control):
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
        kwargs_text, kwargs_mixed = tool._batch_text(
            operation,
            lambda target: _extra_method_kwargs(target, spec),
            lambda value: _format_dict(typing.cast("dict[str, typing.Any]", value)),
        )
        kwargs_edit = tool._line_edit(kwargs_text, parent=page)
        tool._apply_mixed_line_edit(kwargs_edit, kwargs_mixed)
        kwargs_edit.setObjectName(_method_kwargs_object_name(operation.method_family))
        tool._connect_line_edit_finished(
            kwargs_edit,
            lambda text: _update_current_extra_method_kwargs(
                tool, spec, _dict_from_text(text)
            ),
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
    match control.kind:
        case MethodControlKind.TRANSFORM:
            mode_mixed = tool._batch_is_mixed(
                operation, lambda target: target.method_transform
            )
            mode_combo = tool._combo(
                control.options,
                None if mode_mixed else operation.method_transform,
                _method_transform_update_callback(tool),
                parent=layout.parentWidget(),
                mixed=mode_mixed,
            )
            mode_combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, mode_combo, control.tooltip)
            if not mode_mixed and operation.method_transform == "blend":
                x_mixed = tool._batch_is_mixed(
                    operation, lambda target: target.method_transform_x
                )
                x_combo = tool._combo(
                    _TRANSFORM_COMPONENT_OPTIONS,
                    None if x_mixed else operation.method_transform_x,
                    lambda text: tool._update_current_operation(
                        method_transform_x=text
                    ),
                    parent=layout.parentWidget(),
                    mixed=x_mixed,
                )
                x_combo.setObjectName("figureComposerMethodTransformXCombo")
                y_mixed = tool._batch_is_mixed(
                    operation, lambda target: target.method_transform_y
                )
                y_combo = tool._combo(
                    _TRANSFORM_COMPONENT_OPTIONS,
                    None if y_mixed else operation.method_transform_y,
                    lambda text: tool._update_current_operation(
                        method_transform_y=text
                    ),
                    parent=layout.parentWidget(),
                    mixed=y_mixed,
                )
                y_combo.setObjectName("figureComposerMethodTransformYCombo")
                tool._add_compound_form_row(
                    layout,
                    "Blend",
                    (
                        ("x", x_combo, "Transform used for x coordinates."),
                        ("y", y_combo, "Transform used for y coordinates."),
                    ),
                    "Build a blended transform from separate x and y components.",
                )
            elif not mode_mixed and operation.method_transform == "custom":
                expression_text, expression_mixed = tool._batch_text(
                    operation, lambda target: target.method_transform_expression, str
                )
                expression_edit = tool._line_edit(
                    expression_text,
                    parent=layout.parentWidget(),
                )
                tool._apply_mixed_line_edit(expression_edit, expression_mixed)
                expression_edit.setObjectName(
                    "figureComposerMethodTransformExpressionEdit"
                )
                tool._connect_line_edit_finished(
                    expression_edit,
                    lambda text: tool._update_current_operation(
                        method_transform_expression=text
                    ),
                )
                tool._add_form_row(
                    layout,
                    "Expression",
                    expression_edit,
                    "Python expression for transform=.\n"
                    "Available names: ax, fig, mtransforms.",
                )
                trusted_check = tool._check_box(
                    operation.trusted,
                    _operation_trust_update_callback(tool),
                    parent=layout.parentWidget(),
                )
                trusted_check.setObjectName("figureComposerMethodTransformTrustedCheck")
                tool._add_form_row(
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

            mixed = tool._batch_is_mixed(
                operation,
                arg_value_getter,
            )
            combo = tool._combo(
                control.options,
                None if mixed else str(arg_value_getter(operation)),
                _method_arg_callback(tool, index, spec),
                parent=layout.parentWidget(),
                mixed=mixed,
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.INT_ARG:
            index = _control_arg_index(control)
            mixed = tool._batch_is_mixed(
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
            tool._connect_value_signal(
                spinbox,
                spinbox.valueChanged,
                int,
                _method_arg_update_callback(tool, index),
            )
            tool._add_form_row(
                layout,
                control.label,
                tool._mixed_value_widget(
                    spinbox, mixed=mixed, parent=layout.parentWidget()
                ),
                _numeric_control_tooltip(control, mixed),
            )
        case MethodControlKind.FLOAT_ARG:
            index = _control_arg_index(control)
            mixed = tool._batch_is_mixed(
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
            tool._connect_value_signal(
                spinbox,
                spinbox.valueChanged,
                float,
                _method_arg_update_callback(tool, index),
            )
            tool._add_form_row(
                layout,
                control.label,
                tool._mixed_value_widget(
                    spinbox, mixed=mixed, parent=layout.parentWidget()
                ),
                _numeric_control_tooltip(control, mixed),
            )
        case MethodControlKind.TEXT_ARG:
            index = _control_arg_index(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, ""),
                str,
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _method_arg_update_callback(tool, index),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.LITERAL_ARG:
            index = _control_arg_index(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, control.default),
                _format_literal_value,
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _parsed_method_arg_update_callback(
                    tool,
                    index,
                    _literal_value_from_text,
                ),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.LITERAL_SEQUENCE_ARG:
            index = _control_arg_index(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, ()),
                lambda value: (
                    _format_literal_sequence(typing.cast("Sequence[typing.Any]", value))
                    if isinstance(value, (list, tuple))
                    else repr(value)
                ),
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _parsed_method_arg_update_callback(
                    tool,
                    index,
                    _literal_sequence_from_text,
                ),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.PLOT_DATA_ARGS:
            x_text, x_mixed = tool._batch_text(
                operation,
                lambda target: _plot_x_arg_value(target, spec),
                _format_optional_plot_sequence,
            )
            x_edit = tool._line_edit(x_text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(x_edit, x_mixed)
            x_edit.setObjectName("figureComposerAxesMethodPlotXEdit")
            tool._connect_line_edit_finished(
                x_edit,
                lambda text: _update_current_plot_data_arg(tool, "x", text),
            )
            tool._add_form_row(
                layout,
                "X values",
                x_edit,
                "Optional literal x sequence.\nLeave blank to call ax.plot(y, ...).",
            )
            y_text, y_mixed = tool._batch_text(
                operation,
                lambda target: _plot_y_arg_value(target, spec),
                _format_plot_sequence,
            )
            y_edit = tool._line_edit(y_text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(y_edit, y_mixed)
            y_edit.setObjectName("figureComposerAxesMethodPlotYEdit")
            tool._connect_line_edit_finished(
                y_edit,
                lambda text: _update_current_plot_data_arg(tool, "y", text),
            )
            tool._add_form_row(
                layout,
                "Y values",
                y_edit,
                "Required literal y sequence passed to ax.plot.",
            )
        case MethodControlKind.STRING_TUPLE_ARG:
            index = _control_arg_index(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, ()),
                lambda value: (
                    _format_string_tuple(typing.cast("Sequence[str]", value))
                    if isinstance(value, (list, tuple))
                    else ""
                ),
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _method_string_tuple_arg_update_callback(tool, index),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_PAIR_ARGS:
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_float_pair_args(tool, target, spec),
                _format_pair,
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _method_float_pair_args_update_callback(tool),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.ASPECT_ARG:
            index = _control_arg_index(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_arg_value(target, spec, index, control.default),
                _format_aspect_value,
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _parsed_method_arg_update_callback(
                    tool,
                    index,
                    _aspect_value_from_text,
                ),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.BOOL_ARG_COMBO:
            index = _control_arg_index(control)
            mixed = tool._batch_is_mixed(
                operation,
                lambda target: bool(
                    _method_arg_value(target, spec, index, control.default)
                ),
            )
            combo = tool._combo(
                control.options,
                None
                if mixed
                else str(
                    bool(_method_arg_value(operation, spec, index, control.default))
                ),
                _method_bool_arg_callback(tool, index),
                parent=layout.parentWidget(),
                mixed=mixed,
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
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

            mixed = tool._batch_is_mixed(
                operation,
                kwarg_value_getter,
            )
            if control.none_label is None:
                combo = tool._combo(
                    control.options,
                    None if mixed else str(kwarg_value_getter(operation)),
                    _method_kwarg_callback(tool, key),
                    parent=layout.parentWidget(),
                    mixed=mixed,
                )
            else:
                combo = tool._optional_name_combo(
                    control.options,
                    None if mixed else kwarg_value_getter(operation),
                    control.none_label,
                    _method_optional_kwarg_callback(tool, key),
                    parent=layout.parentWidget(),
                    mixed=mixed,
                )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.BOOL_KWARG_COMBO:
            key = _control_key(control)
            mixed = tool._batch_is_mixed(
                operation,
                lambda target: bool(_method_kwarg_value(target, key, control.default)),
            )
            combo = tool._combo(
                control.options,
                None
                if mixed
                else str(bool(_method_kwarg_value(operation, key, control.default))),
                _method_bool_kwarg_callback(tool, key),
                parent=layout.parentWidget(),
                mixed=mixed,
            )
            combo.setObjectName(control.object_name)
            tool._add_form_row(layout, control.label, combo, control.tooltip)
        case MethodControlKind.INT_KWARG:
            key = _control_key(control)
            if _control_uses_numeric_spinbox(control):
                mixed = tool._batch_is_mixed(
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
                tool._connect_value_signal(
                    spinbox,
                    spinbox.valueChanged,
                    int,
                    _method_kwarg_update_callback(tool, key),
                )
                tool._add_form_row(
                    layout,
                    control.label,
                    tool._mixed_value_widget(
                        spinbox, mixed=mixed, parent=layout.parentWidget()
                    ),
                    _numeric_control_tooltip(control, mixed),
                )
            else:
                text, mixed = tool._batch_text(
                    operation,
                    lambda target: _method_kwarg_value(target, key, control.default),
                    _format_int_value,
                )
                edit = tool._line_edit(text, parent=layout.parentWidget())
                tool._apply_mixed_line_edit(edit, mixed)
                edit.setObjectName(control.object_name)
                tool._connect_line_edit_finished(
                    edit,
                    _parsed_method_kwarg_update_callback(
                        tool,
                        key,
                        _optional_int_from_text,
                    ),
                )
                tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_KWARG:
            key = _control_key(control)
            if _control_uses_numeric_spinbox(control):
                mixed = tool._batch_is_mixed(
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
                tool._connect_value_signal(
                    spinbox,
                    spinbox.valueChanged,
                    float,
                    _method_kwarg_update_callback(tool, key),
                )
                tool._add_form_row(
                    layout,
                    control.label,
                    tool._mixed_value_widget(
                        spinbox, mixed=mixed, parent=layout.parentWidget()
                    ),
                    _numeric_control_tooltip(control, mixed),
                )
            else:
                text, mixed = tool._batch_text(
                    operation,
                    lambda target: _method_kwarg_value(target, key, control.default),
                    _format_float_value,
                )
                edit = tool._line_edit(text, parent=layout.parentWidget())
                tool._apply_mixed_line_edit(edit, mixed)
                edit.setObjectName(control.object_name)
                tool._connect_line_edit_finished(
                    edit,
                    _parsed_method_kwarg_update_callback(
                        tool,
                        key,
                        _optional_float_from_text,
                    ),
                )
                tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.SUBPLOTS_ADJUST_KWARG:
            key = _control_key(control)
            mixed = tool._batch_is_mixed(
                operation,
                lambda target: _method_kwarg_value(
                    target, key, _subplots_adjust_default(tool, key)
                ),
            )
            spinbox = _subplots_adjust_spinbox(
                tool,
                operation,
                key,
                mixed=mixed,
                parent=layout.parentWidget(),
            )
            spinbox.setObjectName(control.object_name)
            tooltip = control.tooltip
            if mixed:
                tooltip += "\nSelected steps have multiple values."
            tool._add_form_row(
                layout,
                control.label,
                tool._mixed_value_widget(
                    spinbox, mixed=mixed, parent=layout.parentWidget()
                ),
                tooltip,
            )
        case MethodControlKind.TEXT_KWARG:
            key = _control_key(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, control.default),
                lambda value: str(value or ""),
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(tool, key, _empty_text_as_none),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.COLOR_KWARG:
            key = _control_key(control)
            color_text, color_mixed = tool._batch_text(
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
            tool._apply_mixed_line_edit(color_edit.line_edit, color_mixed)
            tool._connect_value_signal(
                color_edit,
                color_edit.editingFinished,
                color_edit.text,
                _method_color_kwarg_update_callback(tool, key),
                unchanged_mixed=lambda: tool._line_edit_batch_unchanged(
                    color_edit.line_edit
                ),
            )
            tool._add_form_row(layout, control.label, color_edit, control.tooltip)
        case MethodControlKind.LITERAL_KWARG:
            key = _control_key(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, control.default),
                _format_literal_value,
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(
                    tool,
                    key,
                    _optional_literal_from_text,
                ),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.STRING_TUPLE_KWARG:
            key = _control_key(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, ()),
                lambda value: (
                    _format_string_tuple(typing.cast("Sequence[str]", value))
                    if isinstance(value, (list, tuple))
                    else ""
                ),
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(
                    tool,
                    key,
                    _string_tuple_from_text_or_none,
                ),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)
        case MethodControlKind.FLOAT_PAIR_KWARG:
            key = _control_key(control)
            text, mixed = tool._batch_text(
                operation,
                lambda target: _method_kwarg_value(target, key, control.default),
                _format_pair,
            )
            edit = tool._line_edit(text, parent=layout.parentWidget())
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(control.object_name)
            tool._connect_line_edit_finished(
                edit,
                _parsed_method_kwarg_update_callback(
                    tool,
                    key,
                    _float_pair_from_text,
                ),
            )
            tool._add_form_row(layout, control.label, edit, control.tooltip)


def _method_arg_update_callback(
    tool: FigureComposerTool, index: int
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_method_arg, tool, index)


def _parsed_method_arg_update_callback(
    tool: FigureComposerTool,
    index: int,
    parser: Callable[[typing.Any], typing.Any],
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_method_arg_from_value, tool, index, parser)


def _method_kwarg_update_callback(
    tool: FigureComposerTool, key: str
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_method_kwarg, tool, key)


def _subplots_adjust_kwarg_update_callback(
    tool: FigureComposerTool, key: str
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_subplots_adjust_kwarg, tool, key)


def _method_color_kwarg_update_callback(
    tool: FigureComposerTool, key: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        _update_current_method_kwarg(tool, key, color_kw_value_from_text(text))

    return update


def _method_transform_update_callback(
    tool: FigureComposerTool,
) -> Callable[[str], None]:
    def update(text: str) -> None:
        tool._update_current_operation_rebuild(
            method_transform=text,
            trusted=text == "custom",
        )

    return update


def _operation_trust_update_callback(
    tool: FigureComposerTool,
) -> Callable[[bool], None]:
    def update(checked: bool) -> None:
        tool._update_current_operation(trusted=checked)

    return update


def _parsed_method_kwarg_update_callback(
    tool: FigureComposerTool,
    key: str,
    parser: Callable[[typing.Any], typing.Any],
) -> Callable[[typing.Any], None]:
    return functools.partial(_update_current_method_kwarg_from_value, tool, key, parser)


def _method_string_tuple_arg_update_callback(
    tool: FigureComposerTool, index: int
) -> Callable[[str], None]:
    return functools.partial(_update_current_method_string_tuple_arg, tool, index)


def _method_float_pair_args_update_callback(
    tool: FigureComposerTool,
) -> Callable[[str], None]:
    return functools.partial(_update_current_method_args_from_pair_text, tool)


def _update_current_method_arg_from_value(
    tool: FigureComposerTool,
    index: int,
    parser: Callable[[typing.Any], typing.Any],
    value: typing.Any,
) -> None:
    _update_current_method_arg(tool, index, parser(value))


def _update_current_method_kwarg_from_value(
    tool: FigureComposerTool,
    key: str,
    parser: Callable[[typing.Any], typing.Any],
    value: typing.Any,
) -> None:
    _update_current_method_kwarg(tool, key, parser(value))


def _update_current_method_args_from_pair_text(
    tool: FigureComposerTool, text: str
) -> None:
    _update_current_method_args(tool, _float_pair_from_text(text) or ())


def _empty_text_as_none(text: str) -> str | None:
    return text or None


def _string_tuple_from_text_or_none(text: str) -> tuple[str, ...] | None:
    return _string_tuple_from_text(text) or None


def _method_arg_value(
    operation: FigureOperationState,
    spec: MethodSpec,
    index: int,
    default: typing.Any,
) -> typing.Any:
    args = _method_args(operation, spec)
    return args[index] if index < len(args) else default


def _method_float_pair_args(
    tool: FigureComposerTool, operation: FigureOperationState, spec: MethodSpec
) -> tuple[float, float] | None:
    args = _method_args(operation, spec, tool)
    if len(args) < 2:
        return None
    return float(args[0]), float(args[1])


def _method_kwarg_value(
    operation: FigureOperationState, key: str, default: typing.Any
) -> typing.Any:
    return operation.method_kwargs.get(key, default)


def _controlled_method_kwarg_keys(spec: MethodSpec) -> frozenset[str]:
    keys = {
        control.key
        for control in spec.controls
        if control.key is not None
        and control.kind
        in {
            MethodControlKind.KWARG_COMBO,
            MethodControlKind.BOOL_KWARG_COMBO,
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
    tool: FigureComposerTool, spec: MethodSpec, extra_kwargs: dict[str, typing.Any]
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

    tool._update_operations(update_kwargs)


def _plot_x_arg_value(
    operation: FigureOperationState, spec: MethodSpec
) -> tuple[typing.Any, ...] | None:
    args = _method_args(operation, spec)
    if len(args) < 2:
        return None
    return typing.cast("tuple[typing.Any, ...]", args[0])


def _plot_y_arg_value(
    operation: FigureOperationState, spec: MethodSpec
) -> tuple[typing.Any, ...]:
    args = _method_args(operation, spec)
    if len(args) >= 2:
        return typing.cast("tuple[typing.Any, ...]", args[1])
    if args:
        return typing.cast("tuple[typing.Any, ...]", args[0])
    if spec.default_args:
        return typing.cast("tuple[typing.Any, ...]", spec.default_args[0])
    return ()


def _format_plot_sequence(value: typing.Any) -> str:
    if isinstance(value, (list, tuple)):
        return _format_literal_sequence(value)
    return _format_literal_value(value)


def _format_optional_plot_sequence(value: typing.Any) -> str:
    return "" if value is None else _format_plot_sequence(value)


def _plot_sequence_from_text(text: str) -> tuple[typing.Any, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    return tuple(_literal_sequence_from_text(stripped))


def _update_current_plot_data_arg(
    tool: FigureComposerTool,
    axis: typing.Literal["x", "y"],
    text: str,
) -> None:
    value = None if axis == "x" and not text.strip() else _plot_sequence_from_text(text)

    def update_args(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        spec = _method_spec(operation)
        x_value = _plot_x_arg_value(operation, spec)
        y_value = _plot_y_arg_value(operation, spec)
        if axis == "x":
            x_value = value
        else:
            y_value = typing.cast("tuple[typing.Any, ...]", value)
        args = (y_value,) if x_value is None else (x_value, y_value)
        return operation.model_copy(update={"method_args": args})

    tool._update_operations(update_args)


def _subplots_adjust_default(tool: FigureComposerTool, key: str) -> float:
    figure_window = tool._figure_window
    if figure_window is not None and erlab.interactive.utils.qt_is_valid(figure_window):
        return float(getattr(figure_window.figure.subplotpars, key))
    figure = Figure(
        figsize=tool.tool_status.setup.figsize,
        dpi=tool.tool_status.setup.dpi,
        layout=typing.cast("typing.Any", tool.tool_status.setup.layout),
    )
    return float(getattr(figure.subplotpars, key))


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
    tool: FigureComposerTool,
    operation: FigureOperationState,
    key: str,
    *,
    mixed: bool,
    parent: QtWidgets.QWidget | None,
) -> QtWidgets.QDoubleSpinBox:
    spinbox = QtWidgets.QDoubleSpinBox(parent)
    values = _subplots_adjust_values(tool, operation)
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
    tool._connect_value_signal(
        spinbox,
        spinbox.valueChanged,
        float,
        _subplots_adjust_kwarg_update_callback(tool, key),
    )
    return spinbox


def _subplots_adjust_values(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("left", "bottom", "right", "top", "wspace", "hspace"):
        default = _subplots_adjust_default(tool, key)
        value = _method_kwarg_value(operation, key, default)
        with contextlib.suppress(TypeError, ValueError):
            values[key] = float(value)
            continue
        values[key] = default
    return values


def _method_control_visible(
    operation: FigureOperationState, spec: MethodSpec, control: MethodControlSpec
) -> bool:
    if not _is_layout_engine_method(spec) or control.key is None:
        return True
    return control.key in _layout_engine_kwarg_keys(
        _layout_engine_name(operation, spec)
    )


def _is_layout_engine_method(spec: MethodSpec) -> bool:
    return spec.family == FigureMethodFamily.FIGURE and spec.name == "set_layout_engine"


def _layout_engine_name(operation: FigureOperationState, spec: MethodSpec) -> str:
    args = _method_args(operation, spec)
    if not args:
        return "default"
    layout = args[0]
    return layout if isinstance(layout, str) else "default"


def _layout_engine_kwarg_keys(layout: str) -> frozenset[str]:
    return _LAYOUT_ENGINE_KWARGS.get(layout, frozenset[str]())


def _filter_layout_engine_kwargs(
    args: Sequence[typing.Any], kwargs: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    if not args or not isinstance(args[0], str):
        return kwargs
    allowed_keys = _layout_engine_kwarg_keys(args[0])
    return {key: value for key, value in kwargs.items() if key in allowed_keys}


def _is_subplots_adjust_method(spec: MethodSpec) -> bool:
    return spec.family == FigureMethodFamily.FIGURE and spec.name == "subplots_adjust"


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


def _method_arg_callback(
    tool: FigureComposerTool, index: int, spec: MethodSpec
) -> Callable[[str], None]:
    def update(text: str) -> None:
        if _is_layout_engine_method(spec):
            _update_current_layout_engine(tool, index, text)
            return
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


def _method_optional_kwarg_callback(
    tool: FigureComposerTool, key: str
) -> Callable[[str | None], None]:
    def update(text: str | None) -> None:
        _update_current_method_kwarg(tool, key, text)

    return update


def _update_current_layout_engine(
    tool: FigureComposerTool, index: int, text: str
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

    tool._update_operations(
        update_engine,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


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


def _format_aspect_value(value: typing.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return f"{float(value):g}"
    return _code_args((value,))


def _literal_value_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or "=" in stripped:
        return _dict_from_text(stripped)
    return _literal_from_text(stripped)


def _aspect_value_from_text(text: str) -> str | float | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped in {"auto", "equal"}:
        return stripped
    try:
        value = _literal_from_text(stripped)
    except ValueError:
        return stripped
    if isinstance(value, str):
        return value
    if not isinstance(value, int | float):
        return stripped
    return float(value)


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
    return f"eplt.{spec.name}"


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


def _method_doc_url(spec: MethodSpec) -> str | None:
    if spec.doc_url is not None:
        return spec.doc_url
    doc_name = spec.doc_name or spec.call_name
    match spec.family:
        case FigureMethodFamily.AXES:
            return f"{_MATPLOTLIB_DOC_BASE}/matplotlib.axes.Axes.{doc_name}.html"
        case FigureMethodFamily.FIGURE:
            return f"{_MATPLOTLIB_DOC_BASE}/matplotlib.figure.Figure.{doc_name}.html"
        case FigureMethodFamily.ERLAB:
            return f"{_ERLAB_PLOTTING_DOC_BASE}#erlab.plotting.{doc_name}"
    return None


def _open_method_doc_url(url: str) -> None:
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))


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
        method_args=_default_method_args(tool, spec, axes),
        method_kwargs={},
        method_call_policy=None,
        text_values=(),
        method_transform="data",
        method_transform_x="data",
        method_transform_y="axes",
        method_transform_expression="",
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
        method_args=_default_method_args(tool, spec, axes),
        method_kwargs={},
        method_call_policy=None,
        text_values=(),
        method_transform="data",
        method_transform_x="data",
        method_transform_y="axes",
        method_transform_expression="",
        axes=axes,
    )


def _update_current_method_args(
    tool: FigureComposerTool, args: Sequence[typing.Any]
) -> None:
    tool._update_current_operation(method_args=tuple(args))


def _update_current_method_arg(
    tool: FigureComposerTool, index: int, value: typing.Any
) -> None:
    def update_arg(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        args = list(_method_args(operation, _method_spec(operation)))
        while len(args) <= index:
            args.append(None)
        args[index] = value
        return operation.model_copy(update={"method_args": tuple(args)})

    tool._update_operations(update_arg)


def _update_current_method_string_tuple_arg(
    tool: FigureComposerTool, index: int, text: str
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

    tool._update_operations(update_arg)


def _update_current_method_kwarg(
    tool: FigureComposerTool, key: str, value: typing.Any
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

    tool._update_operations(update_kwarg)


def _update_current_subplots_adjust_kwarg(
    tool: FigureComposerTool, key: str, value: typing.Any
) -> None:
    def update_kwarg(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        defaults = _subplots_adjust_values(tool, operation)
        kwargs = dict(operation.method_kwargs)
        if value is None:
            kwargs.pop(key, None)
        else:
            kwargs[key] = value
        kwargs = normalize_subplots_adjust_kwargs(
            kwargs,
            defaults=defaults,
            changed_key=key,
        )
        return operation.model_copy(update={"method_kwargs": kwargs})

    tool._update_operations(update_kwarg)


def _call_policy_from_label(text: str) -> MethodCallPolicy:
    for policy, label in _CALL_POLICY_LABELS.items():
        if label == text:
            return policy
    return MethodCallPolicy(text)


def _update_current_method_call_policy(
    tool: FigureComposerTool, policy: MethodCallPolicy
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

    tool._update_operations(update_policy)


def _update_current_method_text_values(tool: FigureComposerTool, text: str) -> None:
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

    tool._update_operations(update_text_values)


def _method_args(
    operation: FigureOperationState,
    spec: MethodSpec,
    tool: FigureComposerTool | None = None,
) -> tuple[typing.Any, ...]:
    if operation.method_args:
        return operation.method_args
    if (
        tool is not None
        and spec.family == FigureMethodFamily.AXES
        and spec.name in {"set_xlim", "set_ylim"}
    ):
        return _limit_method_default_args(tool, spec, operation.axes)
    return spec.default_args


def _label_values(operation: FigureOperationState) -> str | list[str]:
    values = list(operation.text_values)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return values


def _method_has_transform_control(spec: MethodSpec) -> bool:
    return any(control.kind == MethodControlKind.TRANSFORM for control in spec.controls)


def _transform_component(
    figure: Figure, axis: Axes, component: str
) -> mtransforms.Transform:
    match component:
        case "data":
            return axis.transData
        case "axes":
            return axis.transAxes
        case "figure":
            return figure.transFigure
        case "dpi":
            return figure.dpi_scale_trans
    raise ValueError(f"Unknown transform component: {component}")


def _transform_component_code(component: str) -> str:
    match component:
        case "data":
            return "ax.transData"
        case "axes":
            return "ax.transAxes"
        case "figure":
            return "fig.transFigure"
        case "dpi":
            return "fig.dpi_scale_trans"
    raise ValueError(f"Unknown transform component: {component}")


def _render_method_transform(
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    figure: Figure,
    axis: Axes,
) -> mtransforms.Transform | None:
    if not _method_has_transform_control(spec):
        return None
    match operation.method_transform:
        case "data":
            return None
        case "axes":
            return axis.transAxes
        case "figure":
            return figure.transFigure
        case "dpi":
            return figure.dpi_scale_trans
        case "xaxis":
            return axis.get_xaxis_transform()
        case "yaxis":
            return axis.get_yaxis_transform()
        case "blend":
            return mtransforms.blended_transform_factory(
                _transform_component(figure, axis, operation.method_transform_x),
                _transform_component(figure, axis, operation.method_transform_y),
            )
        case "custom":
            if not operation.trusted:
                raise ValueError("Custom transform expression is not trusted")
            expression = operation.method_transform_expression.strip()
            if not expression:
                raise ValueError("Custom transform expression is empty")
            namespace = {"ax": axis, "fig": figure, "mtransforms": mtransforms}
            transform = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307
            if not isinstance(transform, mtransforms.Transform):
                raise TypeError("Custom transform expression must return a Transform")
            return transform
    raise ValueError(f"Unknown transform mode: {operation.method_transform}")


def _method_transform_code(
    operation: FigureOperationState, spec: MethodSpec
) -> _RawCode | None:
    if not _method_has_transform_control(spec):
        return None
    match operation.method_transform:
        case "data":
            return None
        case "axes":
            return _RawCode("ax.transAxes")
        case "figure":
            return _RawCode("fig.transFigure")
        case "dpi":
            return _RawCode("fig.dpi_scale_trans")
        case "xaxis":
            return _RawCode("ax.get_xaxis_transform()")
        case "yaxis":
            return _RawCode("ax.get_yaxis_transform()")
        case "blend":
            return _RawCode(
                "mtransforms.blended_transform_factory("
                f"{_transform_component_code(operation.method_transform_x)}, "
                f"{_transform_component_code(operation.method_transform_y)})"
            )
        case "custom":
            if not operation.trusted:
                raise ValueError("Custom transform expression is not trusted")
            expression = operation.method_transform_expression.strip()
            if not expression:
                raise ValueError("Custom transform expression is empty")
            return _RawCode(expression)
    raise ValueError(f"Unknown transform mode: {operation.method_transform}")


def _render_args_kwargs(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    figure: Figure | None = None,
    axis: Axes | None = None,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    args = list(_method_args(operation, spec, tool))
    kwargs = dict(spec.default_kwargs)
    kwargs.update(operation.method_kwargs)
    if _method_has_transform_control(spec):
        kwargs.pop("transform", None)
    if spec.text_values_policy == MethodTextValuesPolicy.POSITIONAL:
        args.append(_label_values(operation))
    elif (
        spec.text_values_policy == MethodTextValuesPolicy.KWARG
        and operation.text_values
    ):
        kwargs.setdefault(spec.text_values_kwarg, list(operation.text_values))
    if axis is not None and figure is not None:
        transform = _render_method_transform(operation, spec, figure=figure, axis=axis)
        if transform is not None:
            kwargs["transform"] = transform
    if _is_layout_engine_method(spec):
        kwargs = _filter_layout_engine_kwargs(args, kwargs)
    elif _is_subplots_adjust_method(spec):
        kwargs = normalize_subplots_adjust_kwargs(
            kwargs,
            defaults=_subplots_adjust_values(tool, operation),
        )
    return tuple(args), kwargs


def _code_args_kwargs(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    spec: MethodSpec,
    *,
    axis_transform: str | None = None,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    args = list(_method_args(operation, spec, tool))
    kwargs = dict(spec.default_kwargs)
    kwargs.update(operation.method_kwargs)
    if _method_has_transform_control(spec):
        kwargs.pop("transform", None)
    if spec.text_values_policy == MethodTextValuesPolicy.POSITIONAL:
        args.append(_label_values(operation))
    elif (
        spec.text_values_policy == MethodTextValuesPolicy.KWARG
        and operation.text_values
    ):
        kwargs.setdefault(spec.text_values_kwarg, list(operation.text_values))
    transform_code = _method_transform_code(operation, spec)
    if axis_transform is not None and transform_code is not None:
        kwargs["transform"] = transform_code
    if _is_layout_engine_method(spec):
        kwargs = _filter_layout_engine_kwargs(args, kwargs)
    elif _is_subplots_adjust_method(spec):
        kwargs = normalize_subplots_adjust_kwargs(
            kwargs,
            defaults=_subplots_adjust_values(tool, operation),
        )
    return tuple(args), kwargs


def _erlab_callable(spec: MethodSpec) -> Callable[..., typing.Any]:
    return typing.cast("Callable[..., typing.Any]", getattr(eplt, spec.call_name))


def _render_method(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    figure: Figure,
    axs: typing.Any,
) -> None:
    spec = _method_spec(operation)
    call_policy = _effective_call_policy(operation, spec)
    axes = None
    if spec.target_domain == MethodTargetDomain.AXES:
        axes = _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)

    match call_policy:
        case MethodCallPolicy.BOUND_EACH_AXIS:
            if axes is None:
                return
            for axis in _iter_axes(axes):
                args, kwargs = _render_args_kwargs(
                    tool, operation, spec, figure=figure, axis=axis
                )
                getattr(axis, spec.call_name)(*args, **kwargs)
        case MethodCallPolicy.AXES_POSITIONAL:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(tool, operation, spec)
            _erlab_callable(spec)(axes, *args, **kwargs)
        case MethodCallPolicy.AX_KEYWORD:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(tool, operation, spec)
            _erlab_callable(spec)(*args, ax=axes, **kwargs)
        case MethodCallPolicy.EACH_AXIS_AX_KEYWORD:
            if axes is None:
                return
            args, kwargs = _render_args_kwargs(tool, operation, spec)
            for axis in _iter_axes(axes):
                _erlab_callable(spec)(*args, ax=axis, **kwargs)
        case MethodCallPolicy.BOUND_FIGURE:
            args, kwargs = _render_args_kwargs(tool, operation, spec)
            getattr(figure, spec.call_name)(*args, **kwargs)
        case MethodCallPolicy.FIG_KEYWORD:
            args, kwargs = _render_args_kwargs(tool, operation, spec)
            _erlab_callable(spec)(*args, fig=figure, **kwargs)
        case MethodCallPolicy.PLAIN_CALL:
            args, kwargs = _render_args_kwargs(tool, operation, spec)
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
                tool, operation, spec, axis_transform="ax.transAxes"
            )
            call = _call_code(f"ax.{spec.call_name}", args, kwargs)
            return [
                f"for ax in {_axes_sequence_code(tool, operation.axes)}:",
                f"    {call}",
            ]
        case MethodCallPolicy.AXES_POSITIONAL:
            axes_code = _axes_code(tool, operation.axes, for_plot_slices=False)
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            args_text = _code_args((_RawCode(axes_code), *args))
            kwargs_text = _code_kwargs(kwargs)
            parts = [part for part in (args_text, kwargs_text) if part]
            return [f"eplt.{spec.call_name}({', '.join(parts)})"]
        case MethodCallPolicy.AX_KEYWORD:
            axes_code = _axes_code(tool, operation.axes, for_plot_slices=False)
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            kwargs["ax"] = _RawCode(axes_code)
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]
        case MethodCallPolicy.EACH_AXIS_AX_KEYWORD:
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            kwargs["ax"] = _RawCode("ax")
            call = f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"
            return [
                f"for ax in {_axes_sequence_code(tool, operation.axes)}:",
                f"    {call}",
            ]
        case MethodCallPolicy.BOUND_FIGURE:
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            return [_call_code(f"fig.{spec.call_name}", args, kwargs)]
        case MethodCallPolicy.FIG_KEYWORD:
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            kwargs["fig"] = _RawCode("fig")
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]
        case MethodCallPolicy.PLAIN_CALL:
            args, kwargs = _code_args_kwargs(tool, operation, spec)
            return [f"eplt.{spec.call_name}({_call_parts(args, kwargs)})"]


def _call_parts(args: Sequence[typing.Any], kwargs: dict[str, typing.Any]) -> str:
    return ", ".join(part for part in (_code_args(args), _code_kwargs(kwargs)) if part)


def _method_requires_transform_import(operation: FigureOperationState) -> bool:
    spec = _method_spec(operation)
    return _method_has_transform_control(spec) and (
        operation.method_transform == "blend"
        or (operation.method_transform == "custom" and operation.trusted)
    )


def _loaded_operation(operation: FigureOperationState) -> FigureOperationState:
    if operation.method_transform == "custom":
        return operation.model_copy(update={"trusted": False})
    return operation


def _required_imports(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> Sequence[str]:
    imports: list[str] = []
    spec = _method_spec(operation)
    if spec.family == FigureMethodFamily.ERLAB:
        imports.append("import erlab.plotting as eplt")
    if _method_requires_transform_import(operation):
        imports.append("import matplotlib.transforms as mtransforms")
    return tuple(imports)


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
    loaded_operation=_loaded_operation,
)
