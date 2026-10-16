import json

import pytest

from crossjit_generator.ir import (
    HostOperation,
    HostProvenRegion,
    ProviderSpecialization,
    SpecializationAttribute,
    build_elementwise_ir,
)
from crossjit_generator.recipe import (
    AliasKind, EffectKind, EffectSchema, ErrorSchema,
    FloatingPointErrorKind, IterationKind, MutationKind, Recipe, SemanticExpr,
)
from crossjit_generator.replay import CanonicalIRReplay, ReplayValidationError


def _ir():
    recipes = {}
    for name in ("multiply", "add"):
        recipes[f"provider.{name}"] = Recipe(
            operation_id=f"provider.{name}", arity=2,
            semantic_expression=SemanticExpr.operation(
                name, SemanticExpr.argument(0), SemanticExpr.argument(1)),
            iteration_kind=IterationKind.ELEMENTWISE,
            effects=EffectSchema(
                EffectKind.PURE, MutationKind.NONE, AliasKind.FRESH, False),
            errors=ErrorSchema(True, FloatingPointErrorKind.PROVIDER_DEFINED),
        )
    region = HostProvenRegion(
        provider="provider", semantic_proof_digest="proof",
        inputs=("a", "b", "c", "d"),
        operations=(
            HostOperation("n0", "provider.multiply", ("a", "b"), "x"),
            HostOperation("n1", "provider.multiply", ("c", "d"), "y"),
            HostOperation("n2", "provider.add", ("x", "y"), "out"),
        ), outputs=("out",), virtual_values=("x", "y"),
        materialized_values=("out",),
    )
    specialization = ProviderSpecialization(
        "contract", "fp64", "contiguous_same_shape", "fpe_fallback",
        (SpecializationAttribute("scalar_mask", 4),),
    )
    return build_elementwise_ir(region, recipes, specialization)


def test_replay_round_trip_preserves_slots_dependencies_and_hash():
    record = CanonicalIRReplay.from_ir(
        _ir(), access_kinds=("CONTIGUOUS", "CONTIGUOUS", "SCALAR", "CONTIGUOUS"),
        iteration_descriptor={
            "iteration_rank_class": "FLAT",
            "outer_count": 1,
            "inner_extent": "runtime_element_count",
        },
    )
    assert record.operation_dependencies == ((0, 1), (2, 3), (4, 5))
    assert record.output_map == (6,)
    restored = CanonicalIRReplay.from_data(json.loads(record.serialize()))
    assert restored.serialize() == record.serialize()
    assert restored.stable_hash() == record.stable_hash()
    assert restored.elementwise_ir.serialize() == record.elementwise_ir.serialize()


def test_replay_rejects_workload_or_runtime_object_identity():
    with pytest.raises(ReplayValidationError, match="identity is forbidden"):
        CanonicalIRReplay.from_ir(
            _ir(),
            access_kinds=("CONTIGUOUS",) * 4,
            iteration_descriptor={"benchmark_name": "forbidden"},
        )


def test_replay_detects_tampered_ir_hash():
    record = CanonicalIRReplay.from_ir(
        _ir(), access_kinds=("CONTIGUOUS",) * 4,
        iteration_descriptor={"inner_extent": "runtime_element_count"},
    ).to_data()
    record["canonical_elementwise_ir_hash"] = "0" * 64
    with pytest.raises(ReplayValidationError, match="hash mismatch"):
        CanonicalIRReplay.from_data(record)
