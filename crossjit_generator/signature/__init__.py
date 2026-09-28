"""Stable optimization signatures and demand-driven observations."""

from .model import (
    ELEMENT_BUCKET_UPPER_BOUNDS,
    GenerationMode,
    GenerationRequest,
    GenerationResolution,
    GenerationSource,
    ObservedSignature,
    ObservedSignatureProfile,
    ObservedSignatures,
    OptimizationSignature,
    SignatureOperation,
    SignatureValidationError,
    optimization_signature_from_ir,
    resolve_generation_requests,
)

__all__ = [
    "ELEMENT_BUCKET_UPPER_BOUNDS",
    "GenerationMode",
    "GenerationRequest",
    "GenerationResolution",
    "GenerationSource",
    "ObservedSignature",
    "ObservedSignatureProfile",
    "ObservedSignatures",
    "OptimizationSignature",
    "SignatureOperation",
    "SignatureValidationError",
    "optimization_signature_from_ir",
    "resolve_generation_requests",
]
