"""Deterministic ElementwiseIR constructed from already-proven Host regions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

from crossjit_generator.recipe import Recipe


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class ElementwiseIRValidationError(ValueError):
    """Raised for malformed inputs or an unsupported ElementwiseIR shape."""


def _identifier(value: str, description: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ElementwiseIRValidationError(
            f"invalid {description}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class HostOperation:
    node_id: str
    operation_id: str
    operands: tuple[str, ...]
    result: str

    def __post_init__(self) -> None:
        _identifier(self.node_id, "Host node identity")
        _identifier(self.operation_id, "operation identity")
        _identifier(self.result, "Host result identity")
        if not isinstance(self.operands, tuple) or not self.operands:
            raise ElementwiseIRValidationError(
                "Host operation operands must be a non-empty tuple")
        for operand in self.operands:
            _identifier(operand, "Host operand identity")


@dataclass(frozen=True, slots=True)
class HostProvenRegion:
    """Read-only conclusions supplied by Host proof authority.

    This type is an input envelope.  Constructing it does not prove legality,
    escape, or materialization; those conclusions must already be supplied by
    the Host-side adapter.
    """

    provider: str
    semantic_proof_digest: str
    inputs: tuple[str, ...]
    operations: tuple[HostOperation, ...]
    outputs: tuple[str, ...]
    virtual_values: tuple[str, ...]
    materialized_values: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.provider, "Provider identity")
        if not isinstance(self.semantic_proof_digest, str) or not self.semantic_proof_digest:
            raise ElementwiseIRValidationError(
                "a Host semantic proof digest is required")
        for collection, description in (
            (self.inputs, "input"),
            (self.outputs, "output"),
            (self.virtual_values, "virtual value"),
            (self.materialized_values, "materialized value"),
        ):
            if not isinstance(collection, tuple):
                raise ElementwiseIRValidationError(
                    f"{description} collection must be a tuple")
            for value in collection:
                _identifier(value, description)
            if len(collection) != len(set(collection)):
                raise ElementwiseIRValidationError(
                    f"duplicate {description} identity")
        if not self.inputs or not self.operations or not self.outputs:
            raise ElementwiseIRValidationError(
                "region requires inputs, operations, and outputs")


@dataclass(frozen=True, slots=True, order=True)
class SpecializationAttribute:
    name: str
    value: str | int | bool

    def __post_init__(self) -> None:
        _identifier(self.name, "specialization attribute")
        if not isinstance(self.value, (str, int, bool)):
            raise ElementwiseIRValidationError(
                "unsupported specialization attribute value")


@dataclass(frozen=True, slots=True)
class ProviderSpecialization:
    """Canonical Generator-facing projection of the runtime contract C_P."""

    contract_digest: str
    dtype_class: str
    layout_class: str
    error_mode: str
    attributes: tuple[SpecializationAttribute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.contract_digest, str) or not self.contract_digest:
            raise ElementwiseIRValidationError("contract digest is required")
        _identifier(self.dtype_class, "dtype class")
        _identifier(self.layout_class, "layout class")
        _identifier(self.error_mode, "error mode")
        if (not isinstance(self.attributes, tuple) or
                any(not isinstance(item, SpecializationAttribute)
                    for item in self.attributes)):
            raise ElementwiseIRValidationError(
                "specialization attributes must be typed objects")
        attributes = tuple(sorted(self.attributes, key=lambda item: item.name))
        names = [item.name for item in attributes]
        if len(names) != len(set(names)):
            raise ElementwiseIRValidationError(
                "specialization attribute names must be unique")
        object.__setattr__(self, "attributes", attributes)

    def to_data(self) -> dict[str, Any]:
        return {
            "attributes": [
                {"name": item.name, "value": item.value}
                for item in self.attributes
            ],
            "contract_digest": self.contract_digest,
            "dtype_class": self.dtype_class,
            "error_mode": self.error_mode,
            "layout_class": self.layout_class,
        }


@dataclass(frozen=True, slots=True)
class ElementwiseOperation:
    node_id: str
    operation_id: str
    operands: tuple[str, ...]
    result: str
    recipe_hash: str
    semantic_expression: dict[str, Any]
    effects: dict[str, Any]
    errors: dict[str, Any]

    def to_data(self) -> dict[str, Any]:
        return {
            "effects": self.effects,
            "errors": self.errors,
            "node_id": self.node_id,
            "operands": list(self.operands),
            "operation_id": self.operation_id,
            "recipe_hash": self.recipe_hash,
            "result": self.result,
            "semantic_expression": self.semantic_expression,
        }


@dataclass(frozen=True, slots=True)
class ElementwiseIR:
    schema_version: int
    provider: str
    host_semantic_proof_digest: str
    provider_contract_digest: str
    inputs: tuple[str, ...]
    operations: tuple[ElementwiseOperation, ...]
    outputs: tuple[str, ...]
    virtual_intermediates: tuple[str, ...]
    materialized_values: tuple[str, ...]
    dtype_class: str
    layout_class: str
    error_mode: str
    specialization_attributes: tuple[SpecializationAttribute, ...]

    def to_data(self) -> dict[str, Any]:
        return {
            "dtype_class": self.dtype_class,
            "error_mode": self.error_mode,
            "host_semantic_proof_digest": self.host_semantic_proof_digest,
            "inputs": list(self.inputs),
            "layout_class": self.layout_class,
            "materialized_values": list(self.materialized_values),
            "operations": [operation.to_data()
                           for operation in self.operations],
            "outputs": list(self.outputs),
            "provider": self.provider,
            "provider_contract_digest": self.provider_contract_digest,
            "schema_version": self.schema_version,
            "specialization_attributes": [
                {"name": item.name, "value": item.value}
                for item in self.specialization_attributes
            ],
            "virtual_intermediates": list(self.virtual_intermediates),
        }

    def serialize(self) -> str:
        return json.dumps(
            self.to_data(), ensure_ascii=True, sort_keys=True,
            separators=(",", ":"))

    def stable_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode("ascii")).hexdigest()


def build_elementwise_ir(
    region: HostProvenRegion,
    recipes: Mapping[str, Recipe],
    specialization: ProviderSpecialization,
) -> ElementwiseIR:
    """Lower supplied Host/Provider conclusions without performing legality.

    Validation here is representational: topological references, recipe arity,
    and the current single-output MVP.  The function does not decide whether a
    region is legal, whether a value escapes, or whether it may be virtual.
    """

    if not isinstance(region, HostProvenRegion):
        raise ElementwiseIRValidationError("region must be Host-proven input")
    if not isinstance(specialization, ProviderSpecialization):
        raise ElementwiseIRValidationError("specialization must represent C_P")
    if len(region.outputs) != 1:
        raise ElementwiseIRValidationError(
            "ElementwiseIR MVP supports exactly one output")
    known = set(region.inputs)
    if len(known) != len(region.inputs):
        raise ElementwiseIRValidationError("duplicate region input")
    operations: list[ElementwiseOperation] = []
    results: list[str] = []
    node_ids: set[str] = set()
    for operation in region.operations:
        if operation.node_id in node_ids:
            raise ElementwiseIRValidationError("duplicate Host node identity")
        node_ids.add(operation.node_id)
        recipe = recipes.get(operation.operation_id)
        if recipe is None:
            raise ElementwiseIRValidationError(
                f"missing atomic recipe: {operation.operation_id}")
        if len(operation.operands) != recipe.arity:
            raise ElementwiseIRValidationError(
                f"arity mismatch for {operation.operation_id}")
        missing = [operand for operand in operation.operands
                   if operand not in known]
        if missing:
            raise ElementwiseIRValidationError(
                f"operation {operation.node_id} has unknown dependency {missing[0]}")
        if operation.result in known:
            raise ElementwiseIRValidationError(
                f"duplicate value identity: {operation.result}")
        recipe_data = recipe.to_data()
        operations.append(ElementwiseOperation(
            node_id=operation.node_id,
            operation_id=operation.operation_id,
            operands=operation.operands,
            result=operation.result,
            recipe_hash=recipe.stable_hash(),
            semantic_expression=recipe_data["semantic_expression"],
            effects=recipe_data["effects"],
            errors=recipe_data["errors"],
        ))
        known.add(operation.result)
        results.append(operation.result)
    if any(output not in known for output in region.outputs):
        raise ElementwiseIRValidationError("region output is not defined")
    result_set = set(results)
    virtual = set(region.virtual_values)
    materialized = set(region.materialized_values)
    if virtual & materialized:
        raise ElementwiseIRValidationError(
            "Host materialization conclusions overlap")
    if virtual | materialized != result_set:
        raise ElementwiseIRValidationError(
            "Host materialization conclusions must partition operation results")
    if not set(region.outputs) <= materialized:
        raise ElementwiseIRValidationError(
            "Host output must be listed as materialized")
    return ElementwiseIR(
        schema_version=1,
        provider=region.provider,
        host_semantic_proof_digest=region.semantic_proof_digest,
        provider_contract_digest=specialization.contract_digest,
        inputs=region.inputs,
        operations=tuple(operations),
        outputs=region.outputs,
        virtual_intermediates=region.virtual_values,
        materialized_values=region.materialized_values,
        dtype_class=specialization.dtype_class,
        layout_class=specialization.layout_class,
        error_mode=specialization.error_mode,
        specialization_attributes=specialization.attributes,
    )
