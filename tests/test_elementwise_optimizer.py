from dataclasses import replace

import pytest

from crossjit_generator.domains.elementwise import (
    ElementwiseOptimizer,
    UnsupportedElementwiseIR,
)
from crossjit_generator.ir import (
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


def _recipes():
    result = {}
    for name in ("add", "subtract", "multiply", "true_divide"):
        recipe = Recipe(
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
        result[recipe.operation_id] = recipe
    return result


def _ir(length=4):
    names = ("add", "multiply", "subtract", "true_divide", "add", "multiply")
    inputs = tuple(chr(ord("a") + index) for index in range(length + 1))
    operations = []
    prior = None
    for index in range(length):
        name = names[index % len(names)]
        result = "out" if index == length - 1 else f"t{index}"
        operations.append(HostOperation(
            f"n{index}", f"provider.{name}",
            (inputs[0] if index == 0 else prior, inputs[index + 1]), result))
        prior = result
    region = HostProvenRegion(
        provider="provider",
        semantic_proof_digest=f"proof-{length}",
        inputs=inputs,
        operations=tuple(operations),
        outputs=("out",),
        virtual_values=tuple(operation.result for operation in operations[:-1]),
        materialized_values=("out",),
    )
    specialization = ProviderSpecialization(
        contract_digest="contract",
        dtype_class="fp64",
        layout_class="contiguous_same_shape",
        error_mode="provider_fpe_fallback",
    )
    return build_elementwise_ir(region, _recipes(), specialization)


def test_candidate_preserves_operation_and_dependency_order():
    ir = _ir(6)
    candidate = ElementwiseOptimizer().legal_candidates(ir)[0]
    assert candidate.elementwise_ir is ir
    assert candidate.operation_signature == tuple(
        operation.operation_id for operation in ir.operations)
    assert [operation.operands for operation in candidate.elementwise_ir.operations] == [
        operation.operands for operation in ir.operations
    ]
    assert candidate.lowering_form == "single_compiler_visible_loop"
    assert "profit" not in candidate.serialize()
    assert "benchmark" not in candidate.serialize()


def test_candidate_signature_is_deterministic():
    first = ElementwiseOptimizer().legal_candidates(_ir(4))[0]
    second = ElementwiseOptimizer().legal_candidates(_ir(4))[0]
    assert first.serialize() == second.serialize()
    assert first.stable_hash() == second.stable_hash()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dtype_class", "float32", "dtype"),
        ("layout_class", "strided", "layout"),
    ),
)
def test_unsupported_specialization_is_rejected(field, value, message):
    ir = replace(_ir(), **{field: value})
    with pytest.raises(UnsupportedElementwiseIR, match=message):
        ElementwiseOptimizer().legal_candidates(ir)


@pytest.mark.parametrize("length", (1, 4, 5, 8, 12))
def test_long_dag_preserves_all_intermediates_and_order(length):
    ir = _ir(length)
    candidate = ElementwiseOptimizer().legal_candidates(ir)[0]
    assert candidate.cost_hints.operation_count == length
    assert candidate.cost_hints.virtual_intermediate_count == length - 1
    assert candidate.elementwise_ir.operations == ir.operations
    assert candidate.elementwise_ir.materialized_values == ("out",)


def test_region_length_outside_supported_envelope_is_rejected():
    ir = _ir(2)
    with pytest.raises(UnsupportedElementwiseIR, match="one through twelve"):
        ElementwiseOptimizer().legal_candidates(
            replace(ir, operations=(), virtual_intermediates=()))
    with pytest.raises(UnsupportedElementwiseIR, match="one through twelve"):
        ElementwiseOptimizer().legal_candidates(_ir(13))
