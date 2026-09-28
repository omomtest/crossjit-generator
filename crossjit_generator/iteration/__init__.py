"""Generic Provider-owned iteration descriptions."""

from .model import (
    IterationDescriptor,
    IterationRankClass,
    OperandAccessKind,
    OperandDescriptor,
    OutputDescriptor,
    IterationDescriptorValidationError,
)

__all__ = [
    "IterationDescriptor",
    "IterationDescriptorValidationError",
    "IterationRankClass",
    "OperandAccessKind",
    "OperandDescriptor",
    "OutputDescriptor",
]
