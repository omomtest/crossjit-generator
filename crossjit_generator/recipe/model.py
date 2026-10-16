"""Deterministic static Provider recipe definitions.

Recipes describe atomic operation semantics (Q_P).  Runtime dtype, shape,
layout, and other specialization state belong to C_P and are intentionally
absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")


class RecipeValidationError(ValueError):
    """Raised when a static Provider recipe is malformed."""


class IterationKind(str, Enum):
    ELEMENTWISE = "elementwise"


class EffectKind(str, Enum):
    PURE = "pure"
    PROVIDER_DEFINED = "provider_defined"


class MutationKind(str, Enum):
    NONE = "none"
    PROVIDER_DEFINED = "provider_defined"


class AliasKind(str, Enum):
    FRESH = "fresh"
    PROVIDER_DEFINED = "provider_defined"


class FloatingPointErrorKind(str, Enum):
    NONE = "none"
    PROVIDER_DEFINED = "provider_defined"


@dataclass(frozen=True, slots=True)
class SemanticExpr:
    """Small structural expression for one atomic operation."""

    operator: str
    operands: tuple["SemanticExpr", ...] = ()
    argument_index: int | None = None

    @classmethod
    def argument(cls, index: int) -> "SemanticExpr":
        return cls(operator="argument", argument_index=index)

    @classmethod
    def operation(cls, operator: str, *operands: "SemanticExpr") -> "SemanticExpr":
        return cls(operator=operator, operands=tuple(operands))

    def validate(self, arity: int) -> set[int]:
        if not _IDENTIFIER.fullmatch(self.operator):
            raise RecipeValidationError(
                f"invalid semantic operator: {self.operator!r}")
        if self.operator == "argument":
            if self.operands:
                raise RecipeValidationError("argument expression has operands")
            if (not isinstance(self.argument_index, int) or
                    isinstance(self.argument_index, bool) or
                    not 0 <= self.argument_index < arity):
                raise RecipeValidationError(
                    f"argument index {self.argument_index!r} is outside arity {arity}")
            return {self.argument_index}
        if not isinstance(self.operands, tuple):
            raise RecipeValidationError("semantic operands must be a tuple")
        if self.argument_index is not None:
            raise RecipeValidationError(
                "non-argument expression has an argument index")
        if not self.operands:
            raise RecipeValidationError(
                f"semantic operation {self.operator!r} has no operands")
        used: set[int] = set()
        for operand in self.operands:
            if not isinstance(operand, SemanticExpr):
                raise RecipeValidationError("semantic operand is not an expression")
            used.update(operand.validate(arity))
        return used

    def to_data(self) -> dict[str, Any]:
        if self.operator == "argument":
            return {"argument": self.argument_index}
        return {
            "operator": self.operator,
            "operands": [operand.to_data() for operand in self.operands],
        }


@dataclass(frozen=True, slots=True)
class EffectSchema:
    effect: EffectKind
    mutation: MutationKind
    result_alias: AliasKind
    may_call_host: bool

    def __post_init__(self) -> None:
        if not isinstance(self.effect, EffectKind):
            raise RecipeValidationError("invalid effect kind")
        if not isinstance(self.mutation, MutationKind):
            raise RecipeValidationError("invalid mutation kind")
        if not isinstance(self.result_alias, AliasKind):
            raise RecipeValidationError("invalid alias kind")
        if not isinstance(self.may_call_host, bool):
            raise RecipeValidationError("may_call_host must be boolean")

    def to_data(self) -> dict[str, Any]:
        return {
            "effect": self.effect.value,
            "may_call_host": self.may_call_host,
            "mutation": self.mutation.value,
            "result_alias": self.result_alias.value,
        }


@dataclass(frozen=True, slots=True)
class ErrorSchema:
    may_report: bool
    floating_point: FloatingPointErrorKind
    preserve_operation_order: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.may_report, bool):
            raise RecipeValidationError("may_report must be boolean")
        if not isinstance(self.floating_point, FloatingPointErrorKind):
            raise RecipeValidationError("invalid floating-point error kind")
        if not isinstance(self.preserve_operation_order, bool):
            raise RecipeValidationError(
                "preserve_operation_order must be boolean")

    def to_data(self) -> dict[str, Any]:
        return {
            "floating_point": self.floating_point.value,
            "may_report": self.may_report,
            "preserve_operation_order": self.preserve_operation_order,
        }


ConstraintValue = str | int | bool


@dataclass(frozen=True, slots=True, order=True)
class Constraint:
    name: str
    value: ConstraintValue

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.name):
            raise RecipeValidationError(f"invalid constraint name: {self.name!r}")
        if (not isinstance(self.value, (str, int, bool)) or
                isinstance(self.value, float)):
            raise RecipeValidationError(
                f"unsupported constraint value: {self.value!r}")

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class Recipe:
    operation_id: str
    arity: int
    semantic_expression: SemanticExpr
    iteration_kind: IterationKind
    effects: EffectSchema
    errors: ErrorSchema
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.operation_id):
            raise RecipeValidationError(
                f"invalid operation identity: {self.operation_id!r}")
        if (not isinstance(self.arity, int) or isinstance(self.arity, bool) or
                not 1 <= self.arity <= 255):
            raise RecipeValidationError("arity must be an integer in [1, 255]")
        if not isinstance(self.semantic_expression, SemanticExpr):
            raise RecipeValidationError("semantic_expression must be structural")
        used = self.semantic_expression.validate(self.arity)
        expected = set(range(self.arity))
        if used != expected:
            raise RecipeValidationError(
                f"semantic expression uses arguments {sorted(used)}, "
                f"expected {sorted(expected)}")
        if not isinstance(self.iteration_kind, IterationKind):
            raise RecipeValidationError("invalid iteration kind")
        if not isinstance(self.effects, EffectSchema):
            raise RecipeValidationError("invalid effect schema")
        if not isinstance(self.errors, ErrorSchema):
            raise RecipeValidationError("invalid error schema")
        if (not isinstance(self.constraints, tuple) or
                any(not isinstance(item, Constraint)
                    for item in self.constraints)):
            raise RecipeValidationError("constraints must be Constraint objects")
        constraints = tuple(sorted(self.constraints, key=lambda item: item.name))
        names = [item.name for item in constraints]
        if len(names) != len(set(names)):
            raise RecipeValidationError("constraint names must be unique")
        object.__setattr__(self, "constraints", constraints)

    def to_data(self) -> dict[str, Any]:
        return {
            "arity": self.arity,
            "constraints": [constraint.to_data()
                            for constraint in self.constraints],
            "effects": self.effects.to_data(),
            "errors": self.errors.to_data(),
            "iteration_kind": self.iteration_kind.value,
            "operation_id": self.operation_id,
            "schema_version": SCHEMA_VERSION,
            "semantic_expression": self.semantic_expression.to_data(),
        }

    def serialize(self) -> str:
        return json.dumps(
            self.to_data(), ensure_ascii=True, sort_keys=True,
            separators=(",", ":"))

    def stable_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode("ascii")).hexdigest()
