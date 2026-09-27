import json

import pytest

from crossjit_generator.recipe import (
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
    SemanticExpr,
)


def _recipe(constraints=()):
    return Recipe(
        operation_id="provider.add",
        arity=2,
        semantic_expression=SemanticExpr.operation(
            "add", SemanticExpr.argument(0), SemanticExpr.argument(1)),
        iteration_kind=IterationKind.ELEMENTWISE,
        effects=EffectSchema(
            effect=EffectKind.PURE,
            mutation=MutationKind.NONE,
            result_alias=AliasKind.FRESH,
            may_call_host=False,
        ),
        errors=ErrorSchema(
            may_report=True,
            floating_point=FloatingPointErrorKind.PROVIDER_DEFINED,
        ),
        constraints=constraints,
    )


def test_serialization_and_hash_are_stable_and_constraint_order_independent():
    first = _recipe((Constraint("zeta", 2), Constraint("alpha", True)))
    second = _recipe((Constraint("alpha", True), Constraint("zeta", 2)))
    assert first.serialize() == second.serialize()
    assert first.stable_hash() == second.stable_hash()
    assert len(first.stable_hash()) == 64
    data = json.loads(first.serialize())
    assert data["schema_version"] == 1
    assert data["operation_id"] == "provider.add"
    assert data["constraints"] == [
        {"name": "alpha", "value": True},
        {"name": "zeta", "value": 2},
    ]


@pytest.mark.parametrize(
    "change",
    (
        {"operation_id": "Bad Identity"},
        {"arity": 0},
        {"arity": True},
        {"semantic_expression": SemanticExpr.argument(0)},
        {"semantic_expression": SemanticExpr.argument(2)},
        {"iteration_kind": "elementwise"},
        {"constraints": (Constraint("same", 1), Constraint("same", 2))},
        {"constraints": ("not-a-constraint",)},
    ),
)
def test_invalid_recipe_is_rejected(change):
    fields = {
        "operation_id": "provider.add",
        "arity": 2,
        "semantic_expression": SemanticExpr.operation(
            "add", SemanticExpr.argument(0), SemanticExpr.argument(1)),
        "iteration_kind": IterationKind.ELEMENTWISE,
        "effects": EffectSchema(
            EffectKind.PURE, MutationKind.NONE, AliasKind.FRESH, False),
        "errors": ErrorSchema(
            True, FloatingPointErrorKind.PROVIDER_DEFINED),
        "constraints": (),
    }
    fields.update(change)
    with pytest.raises(RecipeValidationError):
        Recipe(**fields)


def test_hash_changes_with_atomic_semantics():
    add = _recipe()
    subtract = Recipe(
        operation_id="provider.subtract",
        arity=2,
        semantic_expression=SemanticExpr.operation(
            "subtract", SemanticExpr.argument(0), SemanticExpr.argument(1)),
        iteration_kind=add.iteration_kind,
        effects=add.effects,
        errors=add.errors,
    )
    assert add.stable_hash() != subtract.stable_hash()
