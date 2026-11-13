"""Bind an authored console calculation to its recorded ImageTool inputs."""

from __future__ import annotations

import typing

import pydantic

from erlab._provenance._graph import CapturedRevision
from erlab.interactive.imagetool._provenance._code import _replace_code_identifiers
from erlab.interactive.imagetool._provenance._model import (
    DerivationEntry,
    FileReplayCall,
    ReplayStep,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
)
from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation


class ConsoleInput(ScriptInput):
    """A live input carries its graph reference across argument renaming."""

    _capture: CapturedRevision = pydantic.PrivateAttr(default_factory=CapturedRevision)


class CalculatedInput(ConsoleInput):
    """A retained calculation follows the inputs of its producing operation."""


class PreparedFileCall(FileReplayCall):
    """A saved file load whose exact checkpoint is the prepared array."""

    steps: tuple[ReplayStep, ...]


class RecordedProgram(ToolProvenanceOperation):
    """A locally recorded Python workflow. Its executor supplies document trust."""

    live_applicable: typing.ClassVar[bool] = False
    spec: ToolProvenanceSpec

    def with_output_name(self, name: str) -> RecordedProgram:
        """Name the output of a captured console expression for publication."""
        original = self.spec.active_name
        if original is None or original == name:
            return self
        replacements = {original: name}
        steps = []
        for step in self.spec.steps:
            operation = step.operation
            if (
                isinstance(operation, ScriptCodeOperation)
                and operation.code is not None
            ):
                operation = operation.model_copy(
                    update={
                        "code": _replace_code_identifiers(operation.code, replacements)
                    }
                )
                step = step.model_copy(update={"operation": operation})
            steps.append(step)
        seed = self.spec.seed_code
        return RecordedProgram(
            spec=self.spec.model_copy(
                update={
                    "active_name": name,
                    "steps": tuple(steps),
                    "seed_code": _replace_code_identifiers(seed, replacements)
                    if seed
                    else None,
                }
            )
        )

    def input_slots(self) -> tuple[str, ...]:
        # The first source uses the graph's primary "data" slot. Console variable
        # names stay in the program's own namespace, including a variable named data.
        return tuple(f"input_{i}" for i in range(1, len(self.spec.replay_inputs)))

    def input_names(self, inputs: typing.Sequence[tuple[str, str]]) -> dict[str, str]:
        return {
            item.name: key
            for item, (_, key) in zip(self.spec.replay_inputs, inputs, strict=True)
        }

    def derivation_entry(self) -> DerivationEntry:
        return DerivationEntry(self.spec.start_label or "Run console calculation", None)
