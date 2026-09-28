import json
from dataclasses import replace

import pytest

from crossjit_generator.decision import (
    ArtifactDecision,
    DecisionDB,
    DecisionEnvironment,
    DecisionRecord,
    MeasurementPoint,
    MeasurementProvenance,
    ProfitabilityEnvelope,
    RuntimeDecision,
)
from crossjit_generator.signature import OptimizationSignature, SignatureOperation


RECIPE_A = "a" * 64
RECIPE_B = "b" * 64


def signature():
    return OptimizationSignature(
        provider="numpy", provider_abi_version=3, domain="elementwise",
        input_count=3,
        operations=(
            SignatureOperation("numpy.add", (0, 1), RECIPE_A),
            SignatureOperation("numpy.multiply", (3, 2), RECIPE_B),
        ),
        dtype_class="float64", layout_class="contiguous_same_shape",
        error_mode="numpy_fpe_fallback",
        materialization_class="virtual_intermediates_single_output",
    )


def environment():
    return DecisionEnvironment(
        provider="numpy", provider_abi_version=3,
        recipe_hashes=(RECIPE_A, RECIPE_B),
        generator_version="0.2.0", optimizer_version="1",
        backend_version="1", target_triple="x86_64-linux-gnu",
        target_features=("sse2",), compiler_id="gcc",
        compiler_version="13.3.0",
        compiler_flags=("-O3", "-ffp-contract=off"),
    )


def point(low, high, decision):
    return MeasurementPoint(
        low, high, 100.0, 50.0, 120.0, 2.0, 0.833,
        0.01, 0.99, decision, "generic signature measurement")


def record():
    return DecisionRecord(
        signature(), environment(), ArtifactDecision.AUTO_SELECTED,
        MeasurementProvenance(
            "generic_signature_microbenchmark", 5, 7,
            "2026-09-28T00:00:00Z"),
        ProfitabilityEnvelope((
            point(16, 1024, RuntimeDecision.ORIGINAL),
            point(1025, 65536, RuntimeDecision.GENERATED),
        )),
        1024, 512, 2048, "c" * 64, 0.99,
        "stable generated speedup above threshold",
    )


def test_decision_db_round_trip_and_runtime_envelope_are_deterministic():
    database = DecisionDB((record(),))
    restored = DecisionDB.from_data(json.loads(database.serialize()))
    assert restored.serialize() == database.serialize()
    assert restored.runtime_decision(
        signature(), environment(), 64) is RuntimeDecision.ORIGINAL
    assert restored.runtime_decision(
        signature(), environment(), 4096) is RuntimeDecision.GENERATED
    assert restored.runtime_decision(
        signature(), environment(), 1_000_000) is RuntimeDecision.ORIGINAL


@pytest.mark.parametrize("changed", [
    {"recipe_hashes": (RECIPE_A, "d" * 64)},
    {"generator_version": "0.2.1"},
    {"optimizer_version": "2"},
    {"backend_version": "2"},
    {"compiler_id": "clang"},
    {"compiler_version": "18.1.0"},
    {"compiler_flags": ("-O2",)},
    {"target_features": ("avx2",)},
])
def test_environment_change_is_stale_and_never_selected(changed):
    database = DecisionDB((record(),))
    stale_environment = replace(environment(), **changed)
    lookup = database.lookup(signature(), stale_environment)
    assert lookup.state is ArtifactDecision.STALE
    assert lookup.record is None
    assert database.runtime_decision(
        signature(), stale_environment, 4096) is RuntimeDecision.ORIGINAL


def test_signature_change_never_reuses_record():
    changed = replace(signature(), scalar_mask=1)
    database = DecisionDB((record(),))
    assert database.lookup(changed, environment()) is None
    assert database.runtime_decision(
        changed, environment(), 4096) is RuntimeDecision.ORIGINAL


@pytest.mark.parametrize("state", [
    ArtifactDecision.AUTO_REJECTED,
    ArtifactDecision.DISABLED,
    ArtifactDecision.STALE,
])
def test_nonselected_artifact_states_reject_runtime_execution(state):
    rejected = replace(record(), artifact_decision=state)
    assert DecisionDB((rejected,)).runtime_decision(
        signature(), environment(), 4096) is RuntimeDecision.ORIGINAL
