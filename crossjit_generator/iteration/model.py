"""Versioned iteration and operand-access contract for generated loops."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class IterationDescriptorValidationError(ValueError):
    """The Provider iteration projection is malformed or unsupported."""


class IterationRankClass(str, Enum):
    FLAT = "FLAT"
    OUTER_INNER = "OUTER_INNER"


class OperandAccessKind(str, Enum):
    CONTIGUOUS = "CONTIGUOUS"
    SCALAR = "SCALAR"
    ZERO_STRIDE = "ZERO_STRIDE"
    FIXED_STRIDE = "FIXED_STRIDE"


def _extent(value: int | str, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IterationDescriptorValidationError(f"invalid {name}")
    if isinstance(value, int) and value < 0:
        raise IterationDescriptorValidationError(f"negative {name}")
    if isinstance(value, str) and not value:
        raise IterationDescriptorValidationError(f"empty symbolic {name}")


@dataclass(frozen=True, slots=True, order=True)
class OperandDescriptor:
    logical_input_slot: int
    access_kind: OperandAccessKind
    inner_stride: int | str
    outer_stride: int | str | None = None
    broadcast_relation: str = "NONE"

    def __post_init__(self) -> None:
        if (not isinstance(self.logical_input_slot, int) or
                self.logical_input_slot < 0):
            raise IterationDescriptorValidationError(
                "logical input slot must be non-negative")
        if not isinstance(self.access_kind, OperandAccessKind):
            raise IterationDescriptorValidationError("invalid access kind")
        _extent(self.inner_stride, "inner stride")
        if self.outer_stride is not None:
            _extent(self.outer_stride, "outer stride")
        if not isinstance(self.broadcast_relation, str) or not self.broadcast_relation:
            raise IterationDescriptorValidationError(
                "broadcast relation is required")
        if self.access_kind in (OperandAccessKind.SCALAR,
                                OperandAccessKind.ZERO_STRIDE):
            if self.inner_stride != 0:
                raise IterationDescriptorValidationError(
                    "scalar/zero-stride operand must have zero inner stride")

    def to_data(self) -> dict[str, Any]:
        return {
            "access_kind": self.access_kind.value,
            "broadcast_relation": self.broadcast_relation,
            "inner_stride": self.inner_stride,
            "logical_input_slot": self.logical_input_slot,
            "outer_stride": self.outer_stride,
        }


@dataclass(frozen=True, slots=True)
class OutputDescriptor:
    access_kind: OperandAccessKind
    inner_stride: int | str
    outer_stride: int | str | None
    ownership: str

    def __post_init__(self) -> None:
        if self.access_kind is not OperandAccessKind.CONTIGUOUS:
            raise IterationDescriptorValidationError(
                "v1 generated output must be contiguous")
        _extent(self.inner_stride, "output inner stride")
        if self.outer_stride is not None:
            _extent(self.outer_stride, "output outer stride")
        if self.ownership != "FRESH_PROVIDER_OUTPUT":
            raise IterationDescriptorValidationError(
                "v1 output ownership must be fresh Provider output")

    def to_data(self) -> dict[str, Any]:
        return {
            "access_kind": self.access_kind.value,
            "inner_stride": self.inner_stride,
            "outer_stride": self.outer_stride,
            "ownership": self.ownership,
        }


@dataclass(frozen=True, slots=True)
class IterationDescriptor:
    schema_version: int
    iteration_rank_class: IterationRankClass
    outer_count: int | str
    outer_shape_class: str
    inner_extent: int | str
    operands: tuple[OperandDescriptor, ...]
    output: OutputDescriptor

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise IterationDescriptorValidationError(
                "unsupported IterationDescriptor schema")
        if not isinstance(self.iteration_rank_class, IterationRankClass):
            raise IterationDescriptorValidationError("invalid rank class")
        _extent(self.outer_count, "outer count")
        _extent(self.inner_extent, "inner extent")
        if not isinstance(self.outer_shape_class, str) or not self.outer_shape_class:
            raise IterationDescriptorValidationError(
                "outer shape class is required")
        slots = tuple(item.logical_input_slot for item in self.operands)
        if slots != tuple(range(len(self.operands))):
            raise IterationDescriptorValidationError(
                "operand logical slots must be dense and ordered")
        if (self.iteration_rank_class is IterationRankClass.FLAT and
                self.outer_count != 1):
            raise IterationDescriptorValidationError(
                "flat iteration must have one outer iteration")

    @classmethod
    def contiguous_same_shape(
        cls, input_count: int, scalar_mask: int = 0,
    ) -> "IterationDescriptor":
        if not isinstance(input_count, int) or input_count <= 0:
            raise IterationDescriptorValidationError(
                "positive input count is required")
        if (not isinstance(scalar_mask, int) or scalar_mask < 0 or
                scalar_mask >= 1 << input_count):
            raise IterationDescriptorValidationError("invalid scalar mask")
        return cls(
            1, IterationRankClass.FLAT, 1, "NONE",
            "runtime_element_count",
            tuple(OperandDescriptor(
                index,
                (OperandAccessKind.SCALAR if scalar_mask & (1 << index)
                 else OperandAccessKind.CONTIGUOUS),
                (0 if scalar_mask & (1 << index) else "element_size"),
                None,
                ("SCALAR" if scalar_mask & (1 << index) else "NONE"))
                for index in range(input_count)),
            OutputDescriptor(
                OperandAccessKind.CONTIGUOUS, "element_size", None,
                "FRESH_PROVIDER_OUTPUT"),
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "inner_extent": self.inner_extent,
            "iteration_rank_class": self.iteration_rank_class.value,
            "operands": [item.to_data() for item in self.operands],
            "outer_count": self.outer_count,
            "outer_shape_class": self.outer_shape_class,
            "output": self.output.to_data(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "IterationDescriptor":
        try:
            return cls(
                int(data["schema_version"]),
                IterationRankClass(data["iteration_rank_class"]),
                data["outer_count"], data["outer_shape_class"],
                data["inner_extent"],
                tuple(OperandDescriptor(
                    int(item["logical_input_slot"]),
                    OperandAccessKind(item["access_kind"]),
                    item["inner_stride"], item.get("outer_stride"),
                    item["broadcast_relation"],
                ) for item in data["operands"]),
                OutputDescriptor(
                    OperandAccessKind(data["output"]["access_kind"]),
                    data["output"]["inner_stride"],
                    data["output"].get("outer_stride"),
                    data["output"]["ownership"],
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, IterationDescriptorValidationError):
                raise
            raise IterationDescriptorValidationError(
                f"malformed IterationDescriptor: {error}") from error

    def serialize(self) -> str:
        return json.dumps(self.to_data(), ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"))

    def stable_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode("ascii")).hexdigest()
