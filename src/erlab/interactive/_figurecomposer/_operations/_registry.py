"""Explicit operation spec registry for Figure Composer."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._operations import (
    _bz_overlay,
    _custom_code,
    _line_profile,
    _method,
    _photon_energy,
    _plot_slices,
    _set_palette,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from erlab.interactive._figurecomposer._operations._base import (
        AddStepActionSpec,
        OperationSpec,
    )
    from erlab.interactive._figurecomposer._state import FigureOperationKind

_OPERATION_SPECS = {
    spec.kind: spec
    for spec in (
        _set_palette.SPEC,
        _plot_slices.SPEC,
        _line_profile.SPEC,
        _bz_overlay.SPEC,
        _photon_energy.SPEC,
        _method.SPEC,
        _custom_code.SPEC,
    )
}

_ADD_STEP_ACTIONS = {
    action.action_id: action
    for spec in _OPERATION_SPECS.values()
    for action in spec.add_actions
}


def spec_for(kind: FigureOperationKind) -> OperationSpec:
    return _OPERATION_SPECS[kind]


def spec_for_action(action_id: str) -> AddStepActionSpec:
    return _ADD_STEP_ACTIONS[action_id]


def add_step_actions() -> Iterable[AddStepActionSpec]:
    return _ADD_STEP_ACTIONS.values()
