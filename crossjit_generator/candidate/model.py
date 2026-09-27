"""Legal Generator candidate model, separate from profitability selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from crossjit_generator.ir import ElementwiseIR


@dataclass(frozen=True, slots=True, order=True)
class RuntimeAssumption:
    name: str
    value: str | int | bool

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class CostHints:
    operation_count: int
    input_count: int
    virtual_intermediate_count: int

    def to_data(self) -> dict[str, int]:
        return {
            "input_count": self.input_count,
            "operation_count": self.operation_count,
            "virtual_intermediate_count": self.virtual_intermediate_count,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    schema_version: int
    domain: str
    elementwise_ir: ElementwiseIR
    operation_signature: tuple[str, ...]
    lowering_form: str
    runtime_assumptions: tuple[RuntimeAssumption, ...]
    target_requirements: tuple[str, ...]
    cost_hints: CostHints

    def to_data(self) -> dict[str, Any]:
        return {
            "cost_hints": self.cost_hints.to_data(),
            "domain": self.domain,
            "ir_hash": self.elementwise_ir.stable_hash(),
            "lowering_form": self.lowering_form,
            "operation_signature": list(self.operation_signature),
            "runtime_assumptions": [
                assumption.to_data()
                for assumption in self.runtime_assumptions
            ],
            "schema_version": self.schema_version,
            "target_requirements": list(self.target_requirements),
        }

    def serialize(self) -> str:
        return json.dumps(
            self.to_data(), ensure_ascii=True, sort_keys=True,
            separators=(",", ":"))

    def stable_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode("ascii")).hexdigest()
