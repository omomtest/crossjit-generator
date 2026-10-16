"""Static Provider recipe protocol."""

from .model import (
    AliasKind,
    Constraint,
    EffectKind,
    EffectSchema,
    ErrorSchema,
    FloatingPointErrorKind,
    IterationKind,
    MutationKind,
    Recipe,
    RecipeValidationError,
    SCHEMA_VERSION,
    SemanticExpr,
)

__all__ = [
    "AliasKind",
    "Constraint",
    "EffectKind",
    "EffectSchema",
    "ErrorSchema",
    "FloatingPointErrorKind",
    "IterationKind",
    "MutationKind",
    "Recipe",
    "RecipeValidationError",
    "SCHEMA_VERSION",
    "SemanticExpr",
]
