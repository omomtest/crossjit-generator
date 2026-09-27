import json

import pytest

from crossjit_generator.signature import (
    GenerationMode,
    ObservedSignature,
    ObservedSignatures,
    OptimizationSignature,
    SignatureOperation,
    SignatureValidationError,
)


def _signature(operations=("add", "multiply")):
    return OptimizationSignature(
        provider="numpy", provider_abi_version=3, domain="elementwise",
        input_count=len(operations) + 1,
        operations=tuple(
            SignatureOperation(
                f"numpy.{operation}",
                (0 if index == 0 else len(operations) + index,
                 index + 1),
            )
            for index, operation in enumerate(operations)
        ),
        dtype_class="float64", layout_class="contiguous_same_shape",
        error_mode="numpy_fpe_fallback",
        materialization_class="virtual_intermediates_single_output",
    )


def test_modes_are_explicit_and_stable():
    assert [mode.value for mode in GenerationMode] == [
        "AUTO", "FORCE", "DISABLE"]


def test_signature_serialization_and_hash_are_deterministic():
    first = _signature()
    second = OptimizationSignature.from_data(
        json.loads(first.serialize()))
    assert first == second
    assert first.stable_hash() == second.stable_hash()
    assert "benchmark" not in first.serialize()


def test_observations_merge_without_workload_identity():
    signature = _signature()
    observed = ObservedSignatures([
        ObservedSignature(signature, 1, 10, 16, 256),
        ObservedSignature(signature, 2, 20, 32, 1024),
    ])
    record = observed.observations()[0]
    assert (record.prepare_count, record.execution_count) == (3, 30)
    assert (record.min_elements, record.max_elements) == (16, 1024)
    restored = ObservedSignatures.from_data(json.loads(observed.serialize()))
    assert restored.serialize() == observed.serialize()


def test_invalid_future_reference_is_rejected():
    with pytest.raises(SignatureValidationError, match="future"):
        OptimizationSignature(
            provider="numpy", provider_abi_version=3, domain="elementwise",
            input_count=2,
            operations=(SignatureOperation("numpy.add", (0, 2)),),
            dtype_class="float64", layout_class="contiguous_same_shape",
            error_mode="numpy_fpe_fallback",
            materialization_class="single_output",
        )


def test_hash_tampering_is_rejected():
    observed = ObservedSignatures([
        ObservedSignature(_signature(), 1, 1, 8, 8)])
    data = json.loads(observed.serialize())
    data["observations"][0]["signature_hash"] = "0" * 64
    with pytest.raises(SignatureValidationError, match="hash mismatch"):
        ObservedSignatures.from_data(data)
