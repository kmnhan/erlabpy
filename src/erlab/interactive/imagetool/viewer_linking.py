"""Linking and history decorators for ImageTool viewer widgets."""

from __future__ import annotations

import functools
import inspect
import operator
import typing
import weakref

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from qtpy import QtWidgets

    from erlab.interactive.imagetool.viewer import ImageSlicerArea


def _link_splitters(
    s0: QtWidgets.QSplitter, s1: QtWidgets.QSplitter, reverse: bool = False
) -> None:
    s0.blockSignals(True)
    s1.blockSignals(True)
    sizes = s0.sizes()
    total = sum(sizes)
    if reverse:
        sizes = list(reversed(sizes))
        sizes[0] = s1.sizes()[-1]
    else:
        sizes[0] = s1.sizes()[0]
    if all(x == 0 for x in sizes[1:]) and sizes[0] != total:
        sizes[1:] = [1] * len(sizes[1:])
    try:
        factor = (total - sizes[0]) / sum(sizes[1:])
    except ZeroDivisionError:
        factor = 0.0
    for k in range(1, len(sizes)):
        sizes[k] = round(sizes[k] * factor)
    if reverse:
        sizes = list(reversed(sizes))
    s0.setSizes(sizes)
    s0.blockSignals(False)
    s1.blockSignals(False)


def _sync_splitters(s0: QtWidgets.QSplitter, s1: QtWidgets.QSplitter) -> None:
    s0.splitterMoved.connect(lambda: _link_splitters(s1, s0))
    s1.splitterMoved.connect(lambda: _link_splitters(s0, s1))


# class ItoolGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
def suppress_history(method: Callable | None = None):
    """Ignore history changes made within the decorated method."""

    def my_decorator(method: Callable):
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            area = self.slicer_area if hasattr(self, "slicer_area") else self
            with area.history_suppressed():
                return method(self, *args, **kwargs)

        return wrapped

    if method is not None:
        return my_decorator(method)
    return my_decorator


def record_history(method: Callable | None = None):
    """Log history before calling the decorated method."""

    def my_decorator(method: Callable):
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            area = self.slicer_area if hasattr(self, "slicer_area") else self
            transaction_id = kwargs.pop("__slicer_transaction_id", None)
            keep_pending = kwargs.pop(
                "__slicer_keep_pending", area._history_group_active
            )
            return area.record_history_mutation(
                transaction_id,
                lambda: method(self, *args, **kwargs),
                keep_pending=keep_pending,
            )

        typing.cast("typing.Any", wrapped)._itool_records_history = True
        return wrapped

    if method is not None:
        return my_decorator(method)
    return my_decorator


def link_slicer(
    func: Callable | None = None,
    *,
    indices: bool = False,
    steps: bool = False,
    color: bool = False,
):
    """Sync decorated methods across multiple `ImageSlicerArea` instances.

    Parameters
    ----------
    func
        The method to sync across multiple instances of `ImageSlicerArea`.
    indices
        If `True`, the input argument named `value` given to `func` are interpreted as
        indices, and will be converted to appropriate values for other instances of
        `ImageSlicerArea`. The behavior of this conversion is determined by `steps`. If
        `True`, An input argument named `axis` of type integer must be present in the
        decorated method to determine the axis along which the index is to be changed.
    steps
        If `False`, considers `value` as an absolute index. If `True`, considers `value`
        as a relative value such as the number of steps or bins. See the implementation
        of `SlicerLinkProxy` for more information.
    color
        Boolean whether the decorated method is related to visualization, such as
        colormap control.

    """

    def my_decorator(func: Callable):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # skip sync if already synced
            skip_sync = kwargs.pop("__slicer_skip_sync", False)
            transaction_id = kwargs.pop("__slicer_transaction_id", None)
            keep_pending = kwargs.pop("__slicer_keep_pending", None)
            obj = args[0]
            records_history = getattr(func, "_itool_records_history", False)
            if keep_pending is None:
                keep_pending = obj._history_group_active
            sync_enabled = (
                obj.is_linked
                and not skip_sync
                and obj._link_sync_suppressed == 0
                and obj._linking_proxy is not None
                and (not color or obj._linking_proxy.link_colors)
            )
            if records_history and transaction_id is None and sync_enabled:
                transaction_id = obj.next_linked_history_transaction_id()

            source_dims = tuple(obj.data.dims) if sync_enabled else ()
            call_kwargs = dict(kwargs)
            if records_history:
                call_kwargs["__slicer_transaction_id"] = transaction_id
                call_kwargs["__slicer_keep_pending"] = keep_pending
            out = func(*args, **call_kwargs)
            if sync_enabled:
                all_args = inspect.Signature.from_callable(func).bind(*args, **kwargs)
                all_args.apply_defaults()
                obj = all_args.arguments.pop("self")
                if obj._linking_proxy is not None:  # pragma: no branch
                    obj._linking_proxy.sync(
                        obj,
                        func.__name__,
                        all_args.arguments,
                        source_dims,
                        indices,
                        steps,
                        color,
                        transaction_id,
                        keep_pending,
                    )
            return out

        return wrapped

    if func is not None:
        return my_decorator(func)
    return my_decorator


class SlicerLinkProxy:
    """Internal class for handling linked `ImageSlicerArea` s.

    Parameters
    ----------
    *slicers
        The slicers to link.
    link_colors
        Whether to sync color related changes, by default `True`.

    """

    def __init__(self, *slicers: ImageSlicerArea, link_colors: bool = True) -> None:
        self.link_colors = link_colors

        self._children: weakref.WeakSet[ImageSlicerArea] = weakref.WeakSet()
        for s in slicers:
            self.add(s)

    @property
    def children(self) -> weakref.WeakSet[ImageSlicerArea]:
        return self._children

    @property
    def num_children(self) -> int:
        return len(self.children)

    def unlink_all(self) -> None:
        for s in self.children:
            s._linking_proxy = None
        self.children.clear()

    def add(self, slicer_area: ImageSlicerArea) -> None:
        if slicer_area.is_linked:
            if slicer_area._linking_proxy == self:
                return
            raise ValueError("Already linked to another proxy.")
        self.children.add(slicer_area)
        slicer_area._linking_proxy = self

    def remove(self, slicer_area: ImageSlicerArea) -> None:
        self.children.remove(slicer_area)
        slicer_area._linking_proxy = None

    def sync(
        self,
        source: ImageSlicerArea,
        funcname: str,
        arguments: dict[str, typing.Any],
        source_dims: tuple[Hashable, ...],
        indices: bool,
        steps: bool,
        color: bool,
        transaction_id: str | None,
        keep_pending: bool,
    ) -> None:
        r"""Propagate changes across multiple :class:`ImageSlicerArea`\ s.

        This method is invoked every time a method decorated with :func:`link_slicer` in
        a linked `ImageSlicerArea` is called.

        Parameters
        ----------
        source
            Instance of `ImageSlicerArea` corresponding to the called method.
        funcname
            Name of the called method.
        arguments
            Arguments included in the function call.
        source_dims
            Dimension order on ``source`` before the linked method changed it.
        indices, steps, color
            Arguments given to the decorator. See :func:`link_slicer`

        """
        if color and not self.link_colors:
            return
        for target in self.children.difference({source}):
            converted_args = self.convert_args(
                source,
                target,
                dict(arguments),
                source_dims,
                indices,
                steps,
                transaction_id,
                keep_pending,
            )
            if converted_args is None:
                continue
            getattr(target, funcname)(**converted_args)

    def convert_args(
        self,
        source: ImageSlicerArea,
        target: ImageSlicerArea,
        args: dict[str, typing.Any],
        source_dims: tuple[Hashable, ...],
        indices: bool,
        steps: bool,
        transaction_id: str | None,
        keep_pending: bool,
    ) -> dict[str, typing.Any] | None:
        source_axis_for_index: int | None = None
        if "axis" in args:
            source_axis = self._coerce_axis(args["axis"])
            if source_axis is None:
                return None
            mapped_axis = self.convert_axis(source_dims, target, source_axis)
            if mapped_axis is None:
                return None
            source_axis_for_index = source_axis
            args["axis"] = mapped_axis

        if "ax1" in args and "ax2" in args:
            target_axes: list[int] = []
            for key in ("ax1", "ax2"):
                source_axis = self._coerce_axis(args[key])
                if source_axis is None:
                    return None
                mapped_axis = self.convert_axis(source_dims, target, source_axis)
                if mapped_axis is None:
                    return None
                target_axes.append(mapped_axis)
            args["ax1"], args["ax2"] = target_axes

        if args.get("axes") is not None:
            mapped_axes = self.convert_axes(source_dims, target, args["axes"])
            if mapped_axes is None:
                return None
            args["axes"] = mapped_axes

        if indices:
            index: int | None = args.get("value")

            if index is not None:
                mapped_index_axis: int | None = args.get("axis")

                if source_axis_for_index is None or mapped_index_axis is None:
                    raise ValueError(
                        "Axis argument not found in method decorated "
                        "with the `indices=True` argument"
                    )

                args["value"] = self.convert_index(
                    source,
                    target,
                    source_axis_for_index,
                    mapped_index_axis,
                    index,
                    steps,
                )

        args["__slicer_skip_sync"] = True  # passed onto the decorator
        args["__slicer_transaction_id"] = transaction_id
        args["__slicer_keep_pending"] = keep_pending
        return args

    @staticmethod
    def _coerce_axis(axis: typing.Any) -> int | None:
        try:
            return operator.index(axis)
        except TypeError:
            return None

    @staticmethod
    def convert_axis(
        source_dims: tuple[Hashable, ...],
        target: ImageSlicerArea,
        axis: int,
    ) -> int | None:
        if not 0 <= axis < len(source_dims):
            return None
        dim = source_dims[axis]
        try:
            return target.data.dims.index(dim)
        except ValueError:
            return None

    @classmethod
    def convert_axes(
        cls,
        source_dims: tuple[Hashable, ...],
        target: ImageSlicerArea,
        axes: Iterable[typing.Any],
    ) -> tuple[int, ...] | None:
        converted_axes: list[int] = []
        for axis in axes:
            source_axis = cls._coerce_axis(axis)
            if source_axis is None:
                return None
            target_axis = cls.convert_axis(source_dims, target, source_axis)
            if target_axis is not None and target_axis not in converted_axes:
                converted_axes.append(target_axis)
        if not converted_axes:
            return None
        return tuple(converted_axes)

    @staticmethod
    def convert_index(
        source: ImageSlicerArea,
        target: ImageSlicerArea,
        source_axis: int,
        target_axis: int,
        index: int,
        steps: bool,
    ):
        if steps:
            return round(
                index
                * source.array_slicer.incs_uniform[source_axis]
                / target.array_slicer.incs_uniform[target_axis]
            )
        value = source.array_slicer.value_of_index(source_axis, index, uniform=False)
        new_index: int = target.array_slicer.index_of_value(
            target_axis, float(value), uniform=False
        )
        return new_index
