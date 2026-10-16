"""Canonical, identity-free synthetic replay descriptions."""

from .model import (
    CanonicalIRReplay,
    LogicalInputSlot,
    ReplayValidationError,
)

__all__ = ["CanonicalIRReplay", "LogicalInputSlot", "ReplayValidationError"]
