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


class GenerationSource(str, Enum):
    OBSERVED = "observed"
    VALIDATION = "validation"
    DEVELOPER_SEED = "developer_seed"


ELEMENT_BUCKET_UPPER_BOUNDS = (
    16, 32, 64, 128, 256, 512, 1024, 2048,
    4096, 8192, 16384, 32768, 65536, (1 << 64) - 1,
)


def _identifier(value: str, description: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SignatureValidationError(f"invalid {description}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class SignatureOperation:
    operation_id: str
    operands: tuple[int, ...]
    recipe_hash: str

    def __post_init__(self) -> None:
        _identifier(self.operation_id, "operation identity")
        if (not isinstance(self.recipe_hash, str) or
                not re.fullmatch(r"[0-9a-f]{64}", self.recipe_hash)):
            raise SignatureValidationError("recipe hash must be SHA-256 hex")
        if (not isinstance(self.operands, tuple) or not self.operands or
                any(not isinstance(value, int) or value < 0
                    for value in self.operands)):
            raise SignatureValidationError(
                "operation operands must be non-negative integer slots")

    def to_data(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operands": list(self.operands),
            "recipe_hash": self.recipe_hash,
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
                    recipe_hash=operation["recipe_hash"],
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
class ObservedSignatureProfile:
    signature: OptimizationSignature
    prepare_count: int
    accepted_count: int
    execution_count: int
    generated_hit_count: int
    generic_fallback_count: int
    min_elements: int
    max_elements: int
    element_buckets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.signature, OptimizationSignature):
            raise SignatureValidationError("observation requires a signature")
        for value, name in (
            (self.prepare_count, "prepare count"),
            (self.accepted_count, "accepted count"),
            (self.execution_count, "execution count"),
            (self.generated_hit_count, "generated hit count"),
            (self.generic_fallback_count, "generic fallback count"),
            (self.min_elements, "minimum elements"),
            (self.max_elements, "maximum elements"),
        ):
            if not isinstance(value, int) or value < 0:
                raise SignatureValidationError(f"{name} must be non-negative")
        if self.max_elements < self.min_elements:
            raise SignatureValidationError("element range is reversed")
        if (not isinstance(self.element_buckets, tuple) or
                len(self.element_buckets) != len(ELEMENT_BUCKET_UPPER_BOUNDS) or
                any(not isinstance(value, int) or value < 0
                    for value in self.element_buckets)):
            raise SignatureValidationError("invalid element-count buckets")
        if sum(self.element_buckets) != self.execution_count:
            raise SignatureValidationError(
                "element-count buckets must sum to execution count")

    def to_data(self) -> dict[str, Any]:
        return {
            "accepted_count": self.accepted_count,
            "element_buckets": [
                {"count": count, "upper_bound": upper}
                for upper, count in zip(
                    ELEMENT_BUCKET_UPPER_BOUNDS, self.element_buckets)
            ],
            "execution_count": self.execution_count,
            "generated_hit_count": self.generated_hit_count,
            "generic_fallback_count": self.generic_fallback_count,
            "max_elements": self.max_elements,
            "min_elements": self.min_elements,
            "prepare_count": self.prepare_count,
            "signature": self.signature.to_data(),
            "signature_hash": self.signature.stable_hash(),
        }


ObservedSignature = ObservedSignatureProfile


class ObservedSignatures:
    """Deterministic aggregation with no workload-identity input."""

    SCHEMA_VERSION = 1

    def __init__(self, observations: Iterable[ObservedSignatureProfile] = ()) -> None:
        self._records: dict[str, ObservedSignatureProfile] = {}
        for observation in observations:
            self.merge(observation)

    def merge(self, observation: ObservedSignatureProfile) -> None:
        if not isinstance(observation, ObservedSignatureProfile):
            raise SignatureValidationError("expected ObservedSignatureProfile")
        key = observation.signature.stable_hash()
        prior = self._records.get(key)
        if prior is None:
            self._records[key] = observation
            return
        self._records[key] = ObservedSignatureProfile(
            signature=prior.signature,
            prepare_count=prior.prepare_count + observation.prepare_count,
            accepted_count=prior.accepted_count + observation.accepted_count,
            execution_count=prior.execution_count + observation.execution_count,
            generated_hit_count=(prior.generated_hit_count +
                                 observation.generated_hit_count),
            generic_fallback_count=(prior.generic_fallback_count +
                                    observation.generic_fallback_count),
            min_elements=min(prior.min_elements, observation.min_elements),
            max_elements=max(prior.max_elements, observation.max_elements),
            element_buckets=tuple(
                left + right
                for left, right in zip(
                    prior.element_buckets, observation.element_buckets)
            ),
        )

    def observations(self) -> tuple[ObservedSignatureProfile, ...]:
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
            buckets = record["element_buckets"]
            if [item["upper_bound"] for item in buckets] != list(
                    ELEMENT_BUCKET_UPPER_BOUNDS):
                raise SignatureValidationError("unexpected element buckets")
            observations.append(ObservedSignatureProfile(
                signature=signature,
                prepare_count=record["prepare_count"],
                accepted_count=record["accepted_count"],
                execution_count=record["execution_count"],
                generated_hit_count=record["generated_hit_count"],
                generic_fallback_count=record["generic_fallback_count"],
                min_elements=record["min_elements"],
                max_elements=record["max_elements"],
                element_buckets=tuple(item["count"] for item in buckets),
            ))
        return cls(observations)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    signature: OptimizationSignature
    mode: GenerationMode
    source: GenerationSource
    request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.signature, OptimizationSignature):
            raise SignatureValidationError("generation request needs a signature")
        if not isinstance(self.mode, GenerationMode):
            raise SignatureValidationError("invalid generation mode")
        if not isinstance(self.source, GenerationSource):
            raise SignatureValidationError("invalid generation source")
        _identifier(self.request_id, "generation request identity")


@dataclass(frozen=True, slots=True)
class GenerationResolution:
    generated: tuple[GenerationRequest, ...]
    disabled_hashes: tuple[str, ...]
    observed_count: int
    validation_seed_count: int
    forced_count: int


def resolve_generation_requests(
    requests: Iterable[GenerationRequest],
    *,
    auto_selected_hashes: frozenset[str] | None = None,
) -> GenerationResolution:
    """Resolve a bounded request universe with DISABLE > FORCE > AUTO.

    With no selection set, AUTO materializes the observed/validation universe
    used by G10.  Supplying a selection set is the G12 handoff: AUTO then emits
    only DecisionDB-selected signatures, while FORCE remains diagnostic.
    """

    grouped: dict[str, list[GenerationRequest]] = {}
    observed: set[str] = set()
    validation: set[str] = set()
    forced: set[str] = set()
    for request in requests:
        if not isinstance(request, GenerationRequest):
            raise SignatureValidationError("expected GenerationRequest")
        signature_hash = request.signature.stable_hash()
        grouped.setdefault(signature_hash, []).append(request)
        if request.source is GenerationSource.OBSERVED:
            observed.add(signature_hash)
        if request.source is GenerationSource.VALIDATION:
            validation.add(signature_hash)
        if request.mode is GenerationMode.FORCE:
            forced.add(signature_hash)

    generated = []
    disabled = []
    for signature_hash, choices in grouped.items():
        if any(item.mode is GenerationMode.DISABLE for item in choices):
            disabled.append(signature_hash)
            continue
        force = [item for item in choices if item.mode is GenerationMode.FORCE]
        if force:
            generated.append(sorted(force, key=lambda item: item.request_id)[0])
            continue
        auto = [item for item in choices if item.mode is GenerationMode.AUTO]
        if auto and (auto_selected_hashes is None or
                     signature_hash in auto_selected_hashes):
            generated.append(sorted(auto, key=lambda item: (
                item.source.value, item.request_id))[0])

    generated.sort(key=lambda item: item.signature.stable_hash())
    return GenerationResolution(
        generated=tuple(generated),
        disabled_hashes=tuple(sorted(disabled)),
        observed_count=len(observed),
        validation_seed_count=len(validation),
        forced_count=len(forced),
    )


def optimization_signature_from_ir(
    ir: Any,
    *,
    provider_abi_version: int,
    scalar_mask: int | None = None,
) -> OptimizationSignature:
    """Project deterministic ElementwiseIR into structural signature identity."""

    from crossjit_generator.ir import ElementwiseIR

    if not isinstance(ir, ElementwiseIR):
        raise SignatureValidationError("expected ElementwiseIR")
    if scalar_mask is None:
        scalar_mask = next(
            (int(item.value) for item in ir.specialization_attributes
             if item.name == "scalar_mask"), 0)
    slots = {name: index for index, name in enumerate(ir.inputs)}
    operations = []
    for index, operation in enumerate(ir.operations):
        try:
            operands = tuple(slots[name] for name in operation.operands)
        except KeyError as error:
            raise SignatureValidationError(
                f"unknown IR operand: {error.args[0]}") from error
        operations.append(SignatureOperation(
            operation_id=operation.operation_id,
            operands=operands,
            recipe_hash=operation.recipe_hash,
        ))
        slots[operation.result] = len(ir.inputs) + index
    materialization_class = (
        "virtual_intermediates_single_output"
        if len(ir.outputs) == 1 and
        len(ir.virtual_intermediates) == len(ir.operations) - 1
        else "explicit_materialization"
    )
    return OptimizationSignature(
        provider=ir.provider,
        provider_abi_version=provider_abi_version,
        domain="elementwise",
        input_count=len(ir.inputs),
        operations=tuple(operations),
        dtype_class=ir.dtype_class,
        layout_class=ir.layout_class,
        error_mode=ir.error_mode,
        materialization_class=materialization_class,
        scalar_mask=scalar_mask,
    )
