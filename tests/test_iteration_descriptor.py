import json

import pytest

from crossjit_generator.iteration import (
    IterationDescriptor,
    IterationDescriptorValidationError,
    IterationRankClass,
    OperandAccessKind,
    OperandDescriptor,
    OutputDescriptor,
)


def test_contiguous_same_shape_round_trip_is_deterministic():
    descriptor = IterationDescriptor.contiguous_same_shape(3, scalar_mask=2)
    assert descriptor.iteration_rank_class is IterationRankClass.FLAT
    assert [item.access_kind for item in descriptor.operands] == [
        OperandAccessKind.CONTIGUOUS,
        OperandAccessKind.SCALAR,
        OperandAccessKind.CONTIGUOUS,
    ]
    restored = IterationDescriptor.from_data(json.loads(descriptor.serialize()))
    assert restored.serialize() == descriptor.serialize()
    assert restored.stable_hash() == descriptor.stable_hash()


def test_outer_inner_descriptor_expresses_scalar_and_zero_stride():
    descriptor = IterationDescriptor(
        1, IterationRankClass.OUTER_INNER, "runtime_outer_count",
        "AFFINE_ROWS", "runtime_inner_extent",
        (
            OperandDescriptor(0, OperandAccessKind.CONTIGUOUS,
                              "element_size", "runtime_row_stride"),
            OperandDescriptor(1, OperandAccessKind.SCALAR, 0, 0, "SCALAR"),
            OperandDescriptor(2, OperandAccessKind.ZERO_STRIDE, 0, 0,
                              "BROADCAST_INNER"),
        ),
        OutputDescriptor(OperandAccessKind.CONTIGUOUS, "element_size",
                         "runtime_output_row_stride",
                         "FRESH_PROVIDER_OUTPUT"),
    )
    assert descriptor.operands[1].inner_stride == 0
    assert descriptor.operands[2].broadcast_relation == "BROADCAST_INNER"


def test_scalar_requires_zero_inner_stride():
    with pytest.raises(IterationDescriptorValidationError, match="zero inner"):
        OperandDescriptor(0, OperandAccessKind.SCALAR, 8)
