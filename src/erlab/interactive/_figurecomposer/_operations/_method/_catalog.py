"""Catalog and control schemas for curated Figure Composer method calls.

The method operation package renders direct Matplotlib ``ax.*`` and ``fig.*`` calls
alongside curated ``erlab.plotting`` helpers. This module owns their declarative
catalog and editor control schemas. New method support should usually be added by
editing the spec tables here, not by adding operation-specific branches elsewhere.

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

import dataclasses
import enum
import typing

import matplotlib
import matplotlib.scale

from erlab.interactive._figurecomposer._line_style import (
    LINE_MARKER_OPTIONS,
    LINE_STYLE_DEFAULT_LABEL,
    LINE_STYLE_OPTIONS,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureMethodFamily,
    FigureOperationState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


TICK_PARAMS_DEFAULT_KWARGS: dict[str, typing.Any] = {
    "axis": "both",
    "which": "major",
}
TICK_PARAMS_CONTROLLED_KWARGS = frozenset(
    (
        "axis",
        "which",
        "direction",
        "reset",
        "bottom",
        "top",
        "left",
        "right",
        "labelbottom",
        "labeltop",
        "labelleft",
        "labelright",
        "length",
        "width",
        "pad",
        "labelrotation",
        "labelsize",
        "labelfontfamily",
        "colors",
        "color",
        "labelcolor",
        "zorder",
        "grid_color",
        "grid_alpha",
        "grid_linewidth",
        "grid_linestyle",
    )
)


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


class MethodTargetDomain(enum.StrEnum):
    """Object domain targeted by one curated method operation."""

    AXES = "axes"
    FIGURE = "figure"
    NONE = "none"


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
    OPTIONAL_BOOL_KWARG_COMBO = "optional_bool_kwarg_combo"
    INT_KWARG = "int_kwarg"
    FLOAT_KWARG = "float_kwarg"
    SUBPLOTS_ADJUST_KWARG = "subplots_adjust_kwarg"
    TEXT_KWARG = "text_kwarg"
    LITERAL_KWARG = "literal_kwarg"
    STRING_TUPLE_KWARG = "string_tuple_kwarg"
    FLOAT_PAIR_KWARG = "float_pair_kwarg"
    TRANSFORM = "transform"
    COLOR_KWARG = "color_kwarg"
    TICK_PARAMS = "tick_params"


@dataclasses.dataclass(frozen=True)
class MethodControlSpec:
    kind: MethodControlKind
    label: str
    tooltip: str
    object_name: str
    arg_index: int | None = None
    key: str | None = None
    options: tuple[str, ...] = ()
    option_labels: tuple[str, ...] = ()
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
    label: str,
    index: int,
    object_name: str,
    tooltip: str,
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
            "Literal sequences passed to the selected plotting method.\n"
            "Leave x blank to use default x positions."
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
    option_labels: Sequence[str] = (),
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.KWARG_COMBO,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        options=tuple(options),
        option_labels=tuple(option_labels),
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


def _optional_bool_kwarg_combo(
    label: str,
    key: str,
    object_name: str,
    tooltip: str,
    *,
    none_label: str = "Default",
) -> MethodControlSpec:
    return MethodControlSpec(
        kind=MethodControlKind.OPTIONAL_BOOL_KWARG_COMBO,
        label=label,
        key=key,
        object_name=object_name,
        tooltip=tooltip,
        options=("True", "False"),
        default=None,
        none_label=none_label,
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


_FLATTEN_ORDER_OPTIONS = ("C", "F")


_FLATTEN_ORDER_LABELS = ("C (Row)", "F (Column)")


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


def _label_subplots_text_controls(
    object_name_prefix: str,
    *,
    loc_default: str,
    include_generated_label_controls: bool,
) -> tuple[MethodControlSpec, ...]:
    controls = [
        _kwarg_combo(
            "Location",
            "loc",
            _LABEL_LOCATION_OPTIONS,
            loc_default,
            f"{object_name_prefix}LocCombo",
            "Location of the anchored subplot label.",
        ),
        _float_pair_kwarg(
            "Offset",
            "offset",
            f"{object_name_prefix}OffsetEdit",
            "Label offset in display points as dx, dy.",
            default=(0.0, 0.0),
        ),
        _text_kwarg(
            "Prefix",
            "prefix",
            f"{object_name_prefix}PrefixEdit",
            "Text prepended to each subplot label.",
            default="",
        ),
        _text_kwarg(
            "Suffix",
            "suffix",
            f"{object_name_prefix}SuffixEdit",
            "Text appended to each subplot label.",
            default="",
        ),
    ]
    if include_generated_label_controls:
        controls.extend(
            (
                _bool_kwarg_combo(
                    "Numeric labels",
                    "numeric",
                    f"{object_name_prefix}NumericCombo",
                    "Use numbers instead of letters for generated labels.",
                    default=False,
                ),
                _bool_kwarg_combo(
                    "Capital letters",
                    "capital",
                    f"{object_name_prefix}CapitalCombo",
                    "Use capital letters for generated alphabetic labels.",
                    default=False,
                ),
            )
        )
    controls.extend(
        (
            _kwarg_combo(
                "Font weight",
                "fontweight",
                _FONT_WEIGHT_OPTIONS,
                "normal",
                f"{object_name_prefix}FontWeightCombo",
                "Font weight for subplot labels.",
            ),
            _literal_kwarg(
                "Font size",
                "fontsize",
                f"{object_name_prefix}FontSizeEdit",
                "Matplotlib font size. Use 8 or quoted names such as 'large'.",
            ),
        )
    )
    return tuple(controls)


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
    "errorbar": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="errorbar",
        label="Errorbar",
        tooltip="Runs ax.errorbar on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_args=((0.0, 1.0), (0.0, 1.0)),
        controls=(
            _plot_data_args(),
            _color_kwarg(
                "Color",
                "color",
                "figureComposerAxesMethodErrorbarColorEdit",
                "Matplotlib color for the errorbar line.",
            ),
            _kwarg_combo(
                "Line style",
                "linestyle",
                LINE_STYLE_OPTIONS,
                None,
                "figureComposerAxesMethodErrorbarLineStyleCombo",
                "Matplotlib line style for the errorbar line.",
                none_label=LINE_STYLE_DEFAULT_LABEL,
            ),
            _float_kwarg(
                "Line width",
                "linewidth",
                "figureComposerAxesMethodErrorbarLineWidthSpin",
                "Line width for the errorbar line.",
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
                "figureComposerAxesMethodErrorbarMarkerCombo",
                "Matplotlib marker style for the errorbar line.",
                none_label=LINE_STYLE_DEFAULT_LABEL,
            ),
            _float_kwarg(
                "Marker size",
                "markersize",
                "figureComposerAxesMethodErrorbarMarkerSizeSpin",
                "Marker size for the errorbar line.",
                default=float(matplotlib.rcParams["lines.markersize"]),
                minimum=0.0,
                maximum=1_000_000.0,
                step=0.5,
            ),
            _color_kwarg(
                "Marker face",
                "markerfacecolor",
                "figureComposerAxesMethodErrorbarMarkerFaceColorEdit",
                "Matplotlib marker face color.",
            ),
            _color_kwarg(
                "Marker edge",
                "markeredgecolor",
                "figureComposerAxesMethodErrorbarMarkerEdgeColorEdit",
                "Matplotlib marker edge color.",
            ),
            _float_kwarg(
                "Cap size",
                "capsize",
                "figureComposerAxesMethodErrorbarCapSizeSpin",
                "Errorbar cap size in points.",
                default=float(matplotlib.rcParams["errorbar.capsize"]),
                minimum=0.0,
                maximum=1_000_000.0,
                step=0.5,
            ),
            _float_kwarg(
                "Alpha",
                "alpha",
                "figureComposerAxesMethodErrorbarAlphaSpin",
                "Line opacity between 0 and 1.",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
            ),
            _text_kwarg(
                "Label",
                "label",
                "figureComposerAxesMethodErrorbarLabelEdit",
                "Legend label for this errorbar.",
            ),
            _float_kwarg(
                "Z order",
                "zorder",
                "figureComposerAxesMethodErrorbarZOrderSpin",
                "Drawing order for this errorbar.",
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
                "Lower and upper limits as two comma-separated values.\n"
                "Use None to keep one side unchanged.",
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
                "Lower and upper limits as two comma-separated values.\n"
                "Use None to keep one side unchanged.",
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
    "tick_params": MethodSpec(
        family=FigureMethodFamily.AXES,
        name="tick_params",
        label="Tick parameters",
        tooltip="Runs ax.tick_params on every selected axis.",
        target_domain=MethodTargetDomain.AXES,
        call_policy=MethodCallPolicy.BOUND_EACH_AXIS,
        default_kwargs=TICK_PARAMS_DEFAULT_KWARGS,
        controls=(
            MethodControlSpec(
                kind=MethodControlKind.TICK_PARAMS,
                label="Ticks",
                tooltip="Compact editor for ax.tick_params keyword arguments.",
                object_name="figureComposerAxesMethodTickParamsEditor",
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
        preserves_empty_text=True,
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
                option_labels=_FLATTEN_ORDER_LABELS,
            ),
            *_label_subplots_text_controls(
                "figureComposerERLabLabelSubplots",
                loc_default="upper left",
                include_generated_label_controls=True,
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
                "Dictionary of property values, such as eV=[0, -0.1].",
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
                option_labels=_FLATTEN_ORDER_LABELS,
            ),
            *_label_subplots_text_controls(
                "figureComposerERLabLabelProperties",
                loc_default="upper right",
                include_generated_label_controls=False,
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
                option_labels=_FLATTEN_ORDER_LABELS,
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
                option_labels=_FLATTEN_ORDER_LABELS,
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
                option_labels=_FLATTEN_ORDER_LABELS,
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


def _method_specs(
    family: FigureMethodFamily,
) -> Mapping[str, MethodSpec]:
    return _METHODS_BY_FAMILY[family]


def _method_spec(operation: FigureOperationState) -> MethodSpec:
    methods = _method_specs(operation.method_family)
    if operation.method_name in methods:
        return methods[operation.method_name]
    raise ValueError(
        f"Unsupported {operation.method_family.value} method: {operation.method_name!r}"
    )


def _effective_call_policy(
    operation: FigureOperationState, spec: MethodSpec
) -> MethodCallPolicy:
    if operation.method_call_policy is None:
        return spec.call_policy
    try:
        policy = MethodCallPolicy(operation.method_call_policy)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported call policy for {spec.name}: {operation.method_call_policy!r}"
        ) from exc
    if policy in spec.selectable_call_policies:
        return policy
    raise ValueError(f"Call policy {policy.value!r} is not available for {spec.name}")


def _method_selector_text(spec: MethodSpec) -> str:
    return spec.name


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
