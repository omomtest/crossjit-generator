"""Deterministic DecisionDB with separate build and runtime decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping

from crossjit_generator.signature import OptimizationSignature


class DecisionValidationError(ValueError):
    pass


class ArtifactDecision(str, Enum):
    AUTO_SELECTED = "AUTO_SELECTED"
    AUTO_REJECTED = "AUTO_REJECTED"
    FORCED = "FORCED"
    DISABLED = "DISABLED"
    STALE = "STALE"


class RuntimeDecision(str, Enum):
    GENERATED = "GENERATED"
    GENERIC_FALLBACK = "GENERIC_FALLBACK"
    ORIGINAL = "ORIGINAL"


def _sha256(value: str, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise DecisionValidationError(f"{name} must be SHA-256 hex")
    return value


@dataclass(frozen=True, slots=True)
class DecisionEnvironment:
    provider: str
    provider_abi_version: int
    recipe_hashes: tuple[str, ...]
    generator_version: str
    optimizer_version: str
    backend_version: str
    target_triple: str
    target_features: tuple[str, ...]
    compiler_id: str
    compiler_version: str
    compiler_flags: tuple[str, ...]
    artifact_format_version: int = 1

    def __post_init__(self) -> None:
        if not self.provider or self.provider_abi_version < 1:
            raise DecisionValidationError("invalid Provider environment")
        if not self.recipe_hashes:
            raise DecisionValidationError("recipe hashes are required")
        for value in self.recipe_hashes:
            _sha256(value, "recipe hash")
        for value, name in (
            (self.generator_version, "Generator version"),
            (self.optimizer_version, "optimizer version"),
            (self.backend_version, "backend version"),
            (self.target_triple, "target triple"),
            (self.compiler_id, "compiler ID"),
            (self.compiler_version, "compiler version"),
        ):
            if not isinstance(value, str) or not value:
                raise DecisionValidationError(f"{name} is required")
        if self.artifact_format_version < 1:
            raise DecisionValidationError("invalid artifact format version")
        if len(self.target_features) != len(set(self.target_features)):
            raise DecisionValidationError("duplicate target feature")
        object.__setattr__(self, "target_features", tuple(sorted(
            self.target_features)))

    def to_data(self) -> dict[str, Any]:
        return {
            "artifact_format_version": self.artifact_format_version,
            "backend_version": self.backend_version,
            "compiler_flags": list(self.compiler_flags),
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "generator_version": self.generator_version,
            "optimizer_version": self.optimizer_version,
            "provider": self.provider,
            "provider_abi_version": self.provider_abi_version,
            "recipe_hashes": list(self.recipe_hashes),
            "target_features": list(self.target_features),
            "target_triple": self.target_triple,
        }

    def stable_hash(self) -> str:
        serialized = json.dumps(
            self.to_data(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DecisionEnvironment":
        return cls(
            provider=data["provider"],
            provider_abi_version=data["provider_abi_version"],
            recipe_hashes=tuple(data["recipe_hashes"]),
            generator_version=data["generator_version"],
            optimizer_version=data["optimizer_version"],
            backend_version=data["backend_version"],
            target_triple=data["target_triple"],
            target_features=tuple(data["target_features"]),
            compiler_id=data["compiler_id"],
            compiler_version=data["compiler_version"],
            compiler_flags=tuple(data["compiler_flags"]),
            artifact_format_version=data["artifact_format_version"],
        )


@dataclass(frozen=True, slots=True)
class MeasurementProvenance:
    method: str
    independent_processes: int
    samples_per_process: int
    measured_at_utc: str

    def __post_init__(self) -> None:
        if self.method != "generic_signature_microbenchmark":
            raise DecisionValidationError(
                "only generic signature microbenchmarks are production evidence")
        if self.independent_processes < 1 or self.samples_per_process < 1:
            raise DecisionValidationError("invalid measurement sample count")
        if not self.measured_at_utc:
            raise DecisionValidationError("measurement timestamp is required")

    def to_data(self) -> dict[str, Any]:
        return {
            "independent_processes": self.independent_processes,
            "measured_at_utc": self.measured_at_utc,
            "method": self.method,
            "samples_per_process": self.samples_per_process,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "MeasurementProvenance":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class MeasurementPoint:
    min_elements: int
    max_elements: int
    baseline_median_ns: float
    generated_median_ns: float | None
    generic_median_ns: float | None
    generated_speedup: float | None
    generic_speedup: float | None
    max_cv: float
    confidence: float
    decision: RuntimeDecision
    reason: str

    def __post_init__(self) -> None:
        if self.min_elements < 0 or self.max_elements < self.min_elements:
            raise DecisionValidationError("invalid profitability range")
        if self.baseline_median_ns <= 0:
            raise DecisionValidationError("baseline median must be positive")
        for value in (self.generated_median_ns, self.generic_median_ns,
                      self.generated_speedup, self.generic_speedup):
            if value is not None and value <= 0:
                raise DecisionValidationError("measurement must be positive")
        if self.max_cv < 0 or not 0 <= self.confidence <= 1:
            raise DecisionValidationError("invalid noise/confidence metadata")
        if not isinstance(self.decision, RuntimeDecision) or not self.reason:
            raise DecisionValidationError("runtime decision and reason required")

    def to_data(self) -> dict[str, Any]:
        return {
            "baseline_median_ns": self.baseline_median_ns,
            "confidence": self.confidence,
            "decision": self.decision.value,
            "generated_median_ns": self.generated_median_ns,
            "generated_speedup": self.generated_speedup,
            "generic_median_ns": self.generic_median_ns,
            "generic_speedup": self.generic_speedup,
            "max_cv": self.max_cv,
            "max_elements": self.max_elements,
            "min_elements": self.min_elements,
            "reason": self.reason,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "MeasurementPoint":
        values = dict(data)
        values["decision"] = RuntimeDecision(values["decision"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ProfitabilityEnvelope:
    points: tuple[MeasurementPoint, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.points, key=lambda item: item.min_elements))
        for prior, current in zip(ordered, ordered[1:]):
            if current.min_elements <= prior.max_elements:
                raise DecisionValidationError("overlapping profitability ranges")
        object.__setattr__(self, "points", ordered)

    def decide(self, element_count: int) -> RuntimeDecision:
        if element_count < 0:
            return RuntimeDecision.ORIGINAL
        for point in self.points:
            if point.min_elements <= element_count <= point.max_elements:
                return point.decision
        return RuntimeDecision.ORIGINAL

    def to_data(self) -> dict[str, Any]:
        return {"points": [point.to_data() for point in self.points]}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProfitabilityEnvelope":
        return cls(tuple(MeasurementPoint.from_data(item)
                         for item in data["points"]))


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    signature: OptimizationSignature
    environment: DecisionEnvironment
    artifact_decision: ArtifactDecision
    provenance: MeasurementProvenance
    profitability: ProfitabilityEnvelope
    generated_source_bytes: int
    object_bytes: int
    decision_db_bytes: int
    artifact_hash: str | None
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if self.signature.provider != self.environment.provider:
            raise DecisionValidationError("Provider environment mismatch")
        if (self.signature.provider_abi_version !=
                self.environment.provider_abi_version):
            raise DecisionValidationError("Provider ABI environment mismatch")
        signature_recipe_hashes = tuple(
            operation.recipe_hash for operation in self.signature.operations)
        if signature_recipe_hashes != self.environment.recipe_hashes:
            raise DecisionValidationError("recipe environment mismatch")
        if any(value < 0 for value in (
                self.generated_source_bytes, self.object_bytes,
                self.decision_db_bytes)):
            raise DecisionValidationError("code-size values must be non-negative")
        if self.artifact_hash is not None:
            _sha256(self.artifact_hash, "artifact hash")
        if not 0 <= self.confidence <= 1 or not self.reason:
            raise DecisionValidationError("decision confidence/reason required")

    def key(self) -> str:
        return f"{self.signature.stable_hash()}:{self.environment.stable_hash()}"

    def to_data(self) -> dict[str, Any]:
        return {
            "artifact_decision": self.artifact_decision.value,
            "artifact_hash": self.artifact_hash,
            "confidence": self.confidence,
            "decision_db_bytes": self.decision_db_bytes,
            "environment": self.environment.to_data(),
            "generated_source_bytes": self.generated_source_bytes,
            "object_bytes": self.object_bytes,
            "profitability": self.profitability.to_data(),
            "provenance": self.provenance.to_data(),
            "reason": self.reason,
            "signature": self.signature.to_data(),
            "signature_hash": self.signature.stable_hash(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DecisionRecord":
        signature = OptimizationSignature.from_data(data["signature"])
        if signature.stable_hash() != data["signature_hash"]:
            raise DecisionValidationError("DecisionDB signature hash mismatch")
        return cls(
            signature=signature,
            environment=DecisionEnvironment.from_data(data["environment"]),
            artifact_decision=ArtifactDecision(data["artifact_decision"]),
            provenance=MeasurementProvenance.from_data(data["provenance"]),
            profitability=ProfitabilityEnvelope.from_data(data["profitability"]),
            generated_source_bytes=data["generated_source_bytes"],
            object_bytes=data["object_bytes"],
            decision_db_bytes=data["decision_db_bytes"],
            artifact_hash=data["artifact_hash"],
            confidence=data["confidence"],
            reason=data["reason"],
        )


@dataclass(frozen=True, slots=True)
class DecisionLookup:
    state: ArtifactDecision
    record: DecisionRecord | None
    reason: str


class DecisionDB:
    SCHEMA_VERSION = 1

    def __init__(self, records: Iterable[DecisionRecord] = ()) -> None:
        self._records: dict[str, DecisionRecord] = {}
        for record in records:
            if record.key() in self._records:
                raise DecisionValidationError("duplicate DecisionDB record")
            self._records[record.key()] = record

    def records(self) -> tuple[DecisionRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def lookup(self, signature: OptimizationSignature,
               environment: DecisionEnvironment) -> DecisionLookup | None:
        key = f"{signature.stable_hash()}:{environment.stable_hash()}"
        exact = self._records.get(key)
        if exact is not None:
            return DecisionLookup(exact.artifact_decision, exact, "exact match")
        if any(record.signature.stable_hash() == signature.stable_hash()
               for record in self._records.values()):
            return DecisionLookup(
                ArtifactDecision.STALE, None,
                "dependency/environment fingerprint changed")
        return None

    def runtime_decision(self, signature: OptimizationSignature,
                         environment: DecisionEnvironment,
                         element_count: int) -> RuntimeDecision:
        lookup = self.lookup(signature, environment)
        if (lookup is None or lookup.record is None or
                lookup.state is ArtifactDecision.STALE):
            return RuntimeDecision.ORIGINAL
        decision = lookup.record.profitability.decide(element_count)
        if lookup.state is ArtifactDecision.DISABLED:
            return RuntimeDecision.ORIGINAL
        if (decision is RuntimeDecision.GENERATED and
                lookup.state not in (
                    ArtifactDecision.AUTO_SELECTED, ArtifactDecision.FORCED)):
            return RuntimeDecision.ORIGINAL
        return decision

    def to_data(self) -> dict[str, Any]:
        return {
            "records": [record.to_data() for record in self.records()],
            "schema_version": self.SCHEMA_VERSION,
        }

    def serialize(self) -> str:
        return json.dumps(
            self.to_data(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DecisionDB":
        if (set(data) != {"records", "schema_version"} or
                data["schema_version"] != 1):
            raise DecisionValidationError("unsupported DecisionDB schema")
        return cls(DecisionRecord.from_data(item) for item in data["records"])
