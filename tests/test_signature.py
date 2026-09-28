import json

import pytest

from crossjit_generator.signature import (
    ELEMENT_BUCKET_UPPER_BOUNDS,
    GenerationMode,
    GenerationRequest,
    GenerationSource,
    ObservedSignatureProfile,
    ObservedSignatures,
    OptimizationSignature,
    SignatureOperation,
    SignatureValidationError,
    resolve_generation_requests,
)


_RECIPE_HASHES = {
    name: (str(index) * 64)
    for index, name in enumerate(("add", "multiply", "subtract"), 1)
}


def _signature(operations=("add", "multiply"), *, scalar_mask=0,
               layout="contiguous_same_shape"):
    return OptimizationSignature(
        provider="numpy", provider_abi_version=3, domain="elementwise",
        input_count=len(operations) + 1,
        operations=tuple(
            SignatureOperation(
                f"numpy.{operation}",
                (0 if index == 0 else len(operations) + index,
                 index + 1),
                _RECIPE_HASHES[operation],
            )
            for index, operation in enumerate(operations)
        ),
        dtype_class="float64", layout_class=layout,
        error_mode="numpy_fpe_fallback",
        materialization_class="virtual_intermediates_single_output",
        scalar_mask=scalar_mask,
    )


def _buckets(**counts):
    values = [0] * len(ELEMENT_BUCKET_UPPER_BOUNDS)
    for bound, count in counts.items():
        values[ELEMENT_BUCKET_UPPER_BOUNDS.index(int(bound))] = count
    return tuple(values)


def _profile(signature=None, *, prepare=1, accepted=1, execute=3,
             generated=2, fallback=1, minimum=16, maximum=64,
             buckets=None):
    return ObservedSignatureProfile(
        signature or _signature(), prepare, accepted, execute,
        generated, fallback, minimum, maximum,
        buckets or _buckets(**{"16": 1, "32": 1, "64": 1}),
    )


def _request(signature, mode, source, request_id):
    return GenerationRequest(signature, mode, source, request_id)


def test_modes_and_sources_are_explicit_and_stable():
    assert [mode.value for mode in GenerationMode] == [
        "AUTO", "FORCE", "DISABLE"]
    assert [source.value for source in GenerationSource] == [
        "observed", "validation", "developer_seed"]


def test_signature_serialization_hash_and_equality_are_deterministic():
    first = _signature()
    second = OptimizationSignature.from_data(json.loads(first.serialize()))
    assert first == second
    assert first.stable_hash() == second.stable_hash()
    assert len(first.stable_hash()) == 64


@pytest.mark.parametrize("changed", [
    _signature(("add", "subtract")),
    _signature(scalar_mask=2),
    _signature(layout="contiguous_broadcast"),
])
def test_relevant_semantic_differences_change_identity(changed):
    assert changed != _signature()
    assert changed.stable_hash() != _signature().stable_hash()


def test_signature_has_no_benchmark_or_object_identity():
    data = _signature().to_data()
    serialized = _signature().serialize()
    for forbidden in ("benchmark", "workload", "source_path",
                      "npbench", "python_function", "object_id"):
        assert forbidden not in data
        assert forbidden not in serialized.lower()


def test_profiles_merge_counts_ranges_and_buckets_order_independently():
    first = _profile(prepare=1, accepted=1, execute=3, generated=2,
                     fallback=1, minimum=16, maximum=64)
    second = _profile(prepare=2, accepted=2, execute=2, generated=0,
                      fallback=2, minimum=32, maximum=1024,
                      buckets=_buckets(**{"32": 1, "1024": 1}))
    forward = ObservedSignatures((first, second))
    reverse = ObservedSignatures((second, first))
    assert forward.serialize() == reverse.serialize()
    record = forward.observations()[0]
    assert (record.prepare_count, record.accepted_count,
            record.execution_count) == (3, 3, 5)
    assert (record.generated_hit_count,
            record.generic_fallback_count) == (2, 3)
    assert (record.min_elements, record.max_elements) == (16, 1024)
    assert sum(record.element_buckets) == 5
    restored = ObservedSignatures.from_data(json.loads(forward.serialize()))
    assert restored.serialize() == forward.serialize()


def test_auto_force_disable_and_bounded_union_semantics():
    observed = _signature()
    validation = _signature(("add", "subtract"))
    forced = _signature(("multiply", "subtract"))
    requests = (
        _request(observed, GenerationMode.AUTO, GenerationSource.OBSERVED,
                 "observed"),
        _request(validation, GenerationMode.AUTO, GenerationSource.VALIDATION,
                 "validation"),
        _request(forced, GenerationMode.FORCE,
                 GenerationSource.DEVELOPER_SEED, "forced"),
    )
    g10 = resolve_generation_requests(requests)
    assert len(g10.generated) == 3
    assert (g10.observed_count, g10.validation_seed_count,
            g10.forced_count) == (1, 1, 1)
    post_tuning = resolve_generation_requests(
        requests, auto_selected_hashes=frozenset({observed.stable_hash()}))
    assert {item.request_id for item in post_tuning.generated} == {
        "observed", "forced"}


def test_disable_overrides_force_and_auto_regardless_of_order():
    signature = _signature()
    choices = (
        _request(signature, GenerationMode.AUTO, GenerationSource.OBSERVED,
                 "auto"),
        _request(signature, GenerationMode.FORCE,
                 GenerationSource.DEVELOPER_SEED, "force"),
        _request(signature, GenerationMode.DISABLE,
                 GenerationSource.DEVELOPER_SEED, "disable"),
    )
    for requests in (choices, tuple(reversed(choices))):
        resolution = resolve_generation_requests(requests)
        assert resolution.generated == ()
        assert resolution.disabled_hashes == (signature.stable_hash(),)


def test_invalid_future_reference_is_rejected():
    with pytest.raises(SignatureValidationError, match="future"):
        OptimizationSignature(
            provider="numpy", provider_abi_version=3, domain="elementwise",
            input_count=2,
            operations=(SignatureOperation(
                "numpy.add", (0, 2), _RECIPE_HASHES["add"]),),
            dtype_class="float64", layout_class="contiguous_same_shape",
            error_mode="numpy_fpe_fallback",
            materialization_class="single_output",
        )


def test_hash_tampering_is_rejected():
    observed = ObservedSignatures((_profile(),))
    data = json.loads(observed.serialize())
    data["observations"][0]["signature_hash"] = "0" * 64
    with pytest.raises(SignatureValidationError, match="hash mismatch"):
        ObservedSignatures.from_data(data)
