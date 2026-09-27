"""Canonical signatures used for observation, generation, and decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class SignatureValidationError(ValueError):
    """An optimization signature or observation is malformed."""


class GenerationMode(str, Enum):
    AUTO = "AUTO"
    FORCE = "FORCE"
    DISABLE = "DISABLE"


def _identifier(value: str, description: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SignatureValidationError(f"invalid {description}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class SignatureOperation:
    operation_id: str
    operands: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier(self.operation_id, "operation identity")
        if (not isinstance(self.operands, tuple) or not self.operands or
                any(not isinstance(value, int) or value < 0
                    for value in self.operands)):
            raise SignatureValidationError(
                "operation operands must be non-negative integer slots")

    def to_data(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operands": list(self.operands),
        }


@dataclass(frozen=True, slots=True)
class OptimizationSignature:
    provider: str
    provider_abi_version: int
    domain: str
    input_count: int
    operations: tuple[SignatureOperation, ...]
    dtype_class: str
    layout_class: str
    error_mode: str
    materialization_class: str
    scalar_mask: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value, description in (
            (self.provider, "Provider identity"),
            (self.domain, "domain"),
            (self.dtype_class, "dtype class"),
            (self.layout_class, "layout class"),
            (self.error_mode, "error mode"),
            (self.materialization_class, "materialization class"),
        ):
            _identifier(value, description)
        if self.schema_version != 1:
            raise SignatureValidationError("unsupported signature schema")
        if not isinstance(self.provider_abi_version, int) or self.provider_abi_version < 1:
            raise SignatureValidationError("Provider ABI version must be positive")
        if not isinstance(self.input_count, int) or self.input_count < 1:
            raise SignatureValidationError("input count must be positive")
        if not isinstance(self.operations, tuple) or not self.operations:
            raise SignatureValidationError("operations must be a non-empty tuple")
        if not isinstance(self.scalar_mask, int) or self.scalar_mask < 0:
            raise SignatureValidationError("scalar mask must be non-negative")
        if self.scalar_mask >= 1 << self.input_count:
            raise SignatureValidationError("scalar mask exceeds the input count")
        for index, operation in enumerate(self.operations):
            if not isinstance(operation, SignatureOperation):
                raise SignatureValidationError("invalid signature operation")
            available = self.input_count + index
            if any(operand >= available for operand in operation.operands):
                raise SignatureValidationError(
                    "operation references a future or unknown slot")

    def to_data(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "dtype_class": self.dtype_class,
            "error_mode": self.error_mode,
            "input_count": self.input_count,
            "layout_class": self.layout_class,
            "materialization_class": self.materialization_class,
            "operations": [operation.to_data() for operation in self.operations],
            "provider": self.provider,
            "provider_abi_version": self.provider_abi_version,
            "scalar_mask": self.scalar_mask,
            "schema_version": self.schema_version,
        }

    def serialize(self) -> str:
        return json.dumps(
            self.to_data(), ensure_ascii=True, sort_keys=True,
            separators=(",", ":"))

    def stable_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode("ascii")).hexdigest()

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "OptimizationSignature":
        expected = {
            "domain", "dtype_class", "error_mode", "input_count",
            "layout_class", "materialization_class", "operations",
            "provider", "provider_abi_version", "scalar_mask",
            "schema_version",
        }
        if set(data) != expected:
            raise SignatureValidationError("unexpected signature fields")
        return cls(
            provider=data["provider"],
            provider_abi_version=data["provider_abi_version"],
            domain=data["domain"],
            input_count=data["input_count"],
            operations=tuple(
                SignatureOperation(
                    operation_id=operation["operation_id"],
                    operands=tuple(operation["operands"]),
                )
                for operation in data["operations"]
            ),
            dtype_class=data["dtype_class"],
            layout_class=data["layout_class"],
            error_mode=data["error_mode"],
            materialization_class=data["materialization_class"],
            scalar_mask=data["scalar_mask"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ObservedSignature:
    signature: OptimizationSignature
    prepare_count: int
    execution_count: int
    min_elements: int
    max_elements: int

    def __post_init__(self) -> None:
        if not isinstance(self.signature, OptimizationSignature):
            raise SignatureValidationError("observation requires a signature")
        for value, name in (
            (self.prepare_count, "prepare count"),
            (self.execution_count, "execution count"),
            (self.min_elements, "minimum elements"),
            (self.max_elements, "maximum elements"),
        ):
            if not isinstance(value, int) or value < 0:
                raise SignatureValidationError(f"{name} must be non-negative")
        if self.max_elements < self.min_elements:
            raise SignatureValidationError("element range is reversed")

    def to_data(self) -> dict[str, Any]:
        return {
            "execution_count": self.execution_count,
            "max_elements": self.max_elements,
            "min_elements": self.min_elements,
            "prepare_count": self.prepare_count,
            "signature": self.signature.to_data(),
            "signature_hash": self.signature.stable_hash(),
        }


class ObservedSignatures:
    """Deterministic aggregation with no workload-identity input."""

    SCHEMA_VERSION = 1

    def __init__(self, observations: Iterable[ObservedSignature] = ()) -> None:
        self._records: dict[str, ObservedSignature] = {}
        for observation in observations:
            self.merge(observation)

    def merge(self, observation: ObservedSignature) -> None:
        if not isinstance(observation, ObservedSignature):
            raise SignatureValidationError("expected ObservedSignature")
        key = observation.signature.stable_hash()
        prior = self._records.get(key)
        if prior is None:
            self._records[key] = observation
            return
        self._records[key] = ObservedSignature(
            signature=prior.signature,
            prepare_count=prior.prepare_count + observation.prepare_count,
            execution_count=prior.execution_count + observation.execution_count,
            min_elements=min(prior.min_elements, observation.min_elements),
            max_elements=max(prior.max_elements, observation.max_elements),
        )

    def observations(self) -> tuple[ObservedSignature, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def to_data(self) -> dict[str, Any]:
        return {
            "observations": [item.to_data() for item in self.observations()],
            "schema_version": self.SCHEMA_VERSION,
        }

    def serialize(self) -> str:
        return json.dumps(self.to_data(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ObservedSignatures":
        if set(data) != {"observations", "schema_version"} or data["schema_version"] != 1:
            raise SignatureValidationError("unsupported observation document")
        observations = []
        for record in data["observations"]:
            signature = OptimizationSignature.from_data(record["signature"])
            if record["signature_hash"] != signature.stable_hash():
                raise SignatureValidationError("signature hash mismatch")
            observations.append(ObservedSignature(
                signature=signature,
                prepare_count=record["prepare_count"],
                execution_count=record["execution_count"],
                min_elements=record["min_elements"],
                max_elements=record["max_elements"],
            ))
        return cls(observations)
