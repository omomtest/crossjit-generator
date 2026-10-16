"""Serializable invocation model for build-time synthetic IR replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from crossjit_generator.ir import ElementwiseIR


class ReplayValidationError(ValueError):
    """A replay record is malformed or contains runtime identity."""


_FORBIDDEN_KEYS = frozenset((
    "benchmark", "benchmark_name", "workload", "workload_id",
    "python_object", "ndarray_object", "object_id", "source_file",
))


def _plain(value: Any, path: str = "record") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReplayValidationError(f"{path} contains a non-string key")
            if key in _FORBIDDEN_KEYS:
                raise ReplayValidationError(
                    f"runtime/workload identity is forbidden: {key}")
            result[key] = _plain(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ReplayValidationError(f"{path} is not JSON structural data")


@dataclass(frozen=True, slots=True, order=True)
class LogicalInputSlot:
    logical_slot: int
    value_id: str
    access_kind: str

    def to_data(self) -> dict[str, Any]:
        return {
            "access_kind": self.access_kind,
            "logical_slot": self.logical_slot,
            "value_id": self.value_id,
        }


@dataclass(frozen=True, slots=True)
class CanonicalIRReplay:
    """Everything needed to synthesize inputs and invoke canonical IR.

    This is deliberately a build-time structural record.  It may contain no
    Python object identity, ndarray identity, source reconstruction, or
    workload identity.  Host legality is consumed through the persisted IR;
    replay does not attempt to establish it.
    """

    schema_version: int
    elementwise_ir: ElementwiseIR
    logical_inputs: tuple[LogicalInputSlot, ...]
    operation_dependencies: tuple[tuple[int, ...], ...]
    output_map: tuple[int, ...]
    provider_specialization: Mapping[str, Any]
    iteration_descriptor: Mapping[str, Any]
    materialization_assumptions: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ReplayValidationError("unsupported replay schema")
        if not isinstance(self.elementwise_ir, ElementwiseIR):
            raise ReplayValidationError("replay requires canonical ElementwiseIR")
        expected_inputs = tuple(range(len(self.elementwise_ir.inputs)))
        slots = tuple(item.logical_slot for item in self.logical_inputs)
        if slots != expected_inputs:
            raise ReplayValidationError("logical input slots must be dense and ordered")
        if tuple(item.value_id for item in self.logical_inputs) != self.elementwise_ir.inputs:
            raise ReplayValidationError("logical input map disagrees with IR")
        value_slots = {item.value_id: item.logical_slot
                       for item in self.logical_inputs}
        expected_dependencies = []
        next_slot = len(self.logical_inputs)
        for operation in self.elementwise_ir.operations:
            try:
                expected_dependencies.append(tuple(
                    value_slots[value] for value in operation.operands))
            except KeyError as error:
                raise ReplayValidationError(
                    "operation dependency is absent from slot map") from error
            value_slots[operation.result] = next_slot
            next_slot += 1
        if self.operation_dependencies != tuple(expected_dependencies):
            raise ReplayValidationError("operation dependency map disagrees with IR")
        expected_outputs = tuple(value_slots[value]
                                 for value in self.elementwise_ir.outputs)
        if self.output_map != expected_outputs:
            raise ReplayValidationError("output map disagrees with IR")
        _plain(self.provider_specialization, "provider_specialization")
        _plain(self.iteration_descriptor, "iteration_descriptor")
        _plain(self.materialization_assumptions,
               "materialization_assumptions")

    @classmethod
    def from_ir(
        cls,
        ir: ElementwiseIR,
        *,
        access_kinds: tuple[str, ...],
        iteration_descriptor: Mapping[str, Any],
        materialization_assumptions: Mapping[str, Any] | None = None,
    ) -> "CanonicalIRReplay":
        if len(access_kinds) != len(ir.inputs):
            raise ReplayValidationError("one access kind is required per input")
        logical_inputs = tuple(LogicalInputSlot(index, value, access_kinds[index])
                               for index, value in enumerate(ir.inputs))
        value_slots = {value: index for index, value in enumerate(ir.inputs)}
        dependencies = []
        next_slot = len(ir.inputs)
        for operation in ir.operations:
            dependencies.append(tuple(value_slots[value]
                                      for value in operation.operands))
            value_slots[operation.result] = next_slot
            next_slot += 1
        specialization = {
            "attributes": [
                {"name": item.name, "value": item.value}
                for item in ir.specialization_attributes
            ],
            "contract_digest": ir.provider_contract_digest,
            "dtype_class": ir.dtype_class,
            "error_mode": ir.error_mode,
            "layout_class": ir.layout_class,
        }
        materialization = materialization_assumptions or {
            "materialized_values": list(ir.materialized_values),
            "output_ownership": "fresh_provider_output",
            "virtual_intermediates": list(ir.virtual_intermediates),
        }
        return cls(
            1, ir, logical_inputs, tuple(dependencies),
            tuple(value_slots[value] for value in ir.outputs),
            specialization, _plain(iteration_descriptor),
            _plain(materialization),
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "canonical_elementwise_ir": self.elementwise_ir.to_data(),
            "canonical_elementwise_ir_hash": self.elementwise_ir.stable_hash(),
            "iteration_descriptor": _plain(self.iteration_descriptor),
            "logical_input_slots": [item.to_data()
                                    for item in self.logical_inputs],
            "materialization_assumptions": _plain(
                self.materialization_assumptions),
            "operation_dependencies": [list(item)
                                       for item in self.operation_dependencies],
            "output_map": list(self.output_map),
            "provider_specialization": _plain(self.provider_specialization),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CanonicalIRReplay":
        plain = _plain(data)
        try:
            ir = ElementwiseIR.from_data(plain["canonical_elementwise_ir"])
            if ir.stable_hash() != plain["canonical_elementwise_ir_hash"]:
                raise ReplayValidationError("canonical IR hash mismatch")
            return cls(
                int(plain["schema_version"]), ir,
                tuple(LogicalInputSlot(
                    int(item["logical_slot"]), item["value_id"],
                    item["access_kind"])
                    for item in plain["logical_input_slots"]),
                tuple(tuple(int(slot) for slot in dependency)
                      for dependency in plain["operation_dependencies"]),
                tuple(int(slot) for slot in plain["output_map"]),
                plain["provider_specialization"],
                plain["iteration_descriptor"],
                plain["materialization_assumptions"],
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ReplayValidationError):
                raise
            raise ReplayValidationError(f"malformed replay record: {error}") from error

    def serialize(self) -> str:
        return json.dumps(self.to_data(), ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"))

    def stable_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode("ascii")).hexdigest()
