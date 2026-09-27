import pytest

from crossjit_generator.ir import (
    ElementwiseIRValidationError,
    HostOperation,
    HostProvenRegion,
    ProviderSpecialization,
    build_elementwise_ir,
)
from crossjit_generator.recipe import (
    AliasKind,
    EffectKind,
    EffectSchema,
    ErrorSchema,
    FloatingPointErrorKind,
    IterationKind,
    MutationKind,
    Recipe,
    SemanticExpr,
)


def _recipe(name):
    return Recipe(
        operation_id=f"provider.{name}",
        arity=2,
        semantic_expression=SemanticExpr.operation(
            name, SemanticExpr.argument(0), SemanticExpr.argument(1)),
        iteration_kind=IterationKind.ELEMENTWISE,
        effects=EffectSchema(
            EffectKind.PURE, MutationKind.NONE, AliasKind.FRESH, False),
        errors=ErrorSchema(
            True, FloatingPointErrorKind.PROVIDER_DEFINED),
    )


RECIPES = {f"provider.{name}": _recipe(name)
           for name in ("add", "subtract", "multiply", "true_divide")}
SPECIALIZATION = ProviderSpecialization(
    contract_digest="contract-fp64-contiguous",
    dtype_class="fp64",
    layout_class="contiguous_same_shape",
    error_mode="provider_fpe_fallback",
)


def _linear(length):
    names = ("add", "multiply", "subtract", "true_divide", "add", "multiply")
    inputs = tuple(chr(ord("a") + index) for index in range(length + 1))
    operations = []
    prior = None
    for index, name in enumerate(names[:length]):
        result = "out" if index == length - 1 else f"t{index}"
        left = inputs[0] if index == 0 else prior
        right = inputs[index + 1]
        operations.append(HostOperation(
            node_id=f"n{index}", operation_id=f"provider.{name}",
            operands=(left, right), result=result))
        prior = result
    virtual = tuple(operation.result for operation in operations[:-1])
    return HostProvenRegion(
        provider="provider",
        semantic_proof_digest=f"proof-linear-{length}",
        inputs=inputs,
        operations=tuple(operations),
        outputs=("out",),
        virtual_values=virtual,
        materialized_values=("out",),
    )


@pytest.mark.parametrize("length", range(2, 7))
def test_linear_l2_through_l6_are_deterministic(length):
    region = _linear(length)
    first = build_elementwise_ir(region, RECIPES, SPECIALIZATION)
    second = build_elementwise_ir(region, dict(reversed(tuple(RECIPES.items()))),
                                  SPECIALIZATION)
    assert first.serialize() == second.serialize()
    assert first.stable_hash() == second.stable_hash()
    assert len(first.operations) == length
    assert first.virtual_intermediates == tuple(
        operation.result for operation in region.operations[:-1])
    assert first.outputs == ("out",)


def test_branched_dag_preserves_dependencies_and_order():
    region = HostProvenRegion(
        provider="provider",
        semantic_proof_digest="proof-branched",
        inputs=("a", "b", "c", "d"),
        operations=(
            HostOperation("n0", "provider.multiply", ("a", "b"), "t0"),
            HostOperation("n1", "provider.multiply", ("c", "d"), "t1"),
            HostOperation("n2", "provider.add", ("t0", "t1"), "out"),
        ),
        outputs=("out",),
        virtual_values=("t0", "t1"),
        materialized_values=("out",),
    )
    ir = build_elementwise_ir(region, RECIPES, SPECIALIZATION)
    assert [operation.node_id for operation in ir.operations] == ["n0", "n1", "n2"]
    assert ir.operations[2].operands == ("t0", "t1")


def test_invalid_dependency_is_rejected_without_reordering():
    region = HostProvenRegion(
        provider="provider",
        semantic_proof_digest="proof-invalid",
        inputs=("a", "b"),
        operations=(
            HostOperation("n0", "provider.add", ("future", "b"), "out"),
        ),
        outputs=("out",),
        virtual_values=(),
        materialized_values=("out",),
    )
    with pytest.raises(ElementwiseIRValidationError, match="unknown dependency"):
        build_elementwise_ir(region, RECIPES, SPECIALIZATION)


def test_multiple_outputs_are_rejected_by_mvp():
    region = HostProvenRegion(
        provider="provider",
        semantic_proof_digest="proof-multi",
        inputs=("a", "b", "c"),
        operations=(
            HostOperation("n0", "provider.add", ("a", "b"), "x"),
            HostOperation("n1", "provider.multiply", ("x", "c"), "y"),
        ),
        outputs=("x", "y"),
        virtual_values=(),
        materialized_values=("x", "y"),
    )
    with pytest.raises(ElementwiseIRValidationError, match="exactly one output"):
        build_elementwise_ir(region, RECIPES, SPECIALIZATION)
