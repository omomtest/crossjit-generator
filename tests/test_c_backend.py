from __future__ import annotations

from array import array
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from crossjit_generator.backend.c import CBackend, generate_bundle
from crossjit_generator.domains.elementwise import ElementwiseOptimizer
from crossjit_generator.ir import (
    HostOperation,
    HostProvenRegion,
    ProviderSpecialization,
    build_elementwise_ir,
)
from crossjit_generator.iteration import (
    IterationDescriptor, IterationRankClass, OperandAccessKind,
    OperandDescriptor, OutputDescriptor,
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
from crossjit_generator.signature import optimization_signature_from_ir


@pytest.mark.parametrize("outer", [1, 4])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("length", [1, 2, 8, 12])
def test_affine_v2_mixed_access_executes_with_runtime_signed_strides(tmp_path, outer, reverse, length):
    original = _candidate(length=length)
    rank = IterationRankClass.FLAT if outer == 1 else IterationRankClass.OUTER_INNER
    descriptor = IterationDescriptor(
        2, rank, 1 if outer == 1 else "runtime_outer_count",
        "NONE" if outer == 1 else "AFFINE_ROWS", "runtime_inner_extent",
        (OperandDescriptor(0, OperandAccessKind.CONTIGUOUS, "element_size",
                           None if outer == 1 else "runtime_outer_stride"),
         OperandDescriptor(1, OperandAccessKind.FIXED_STRIDE, "runtime_inner_stride",
                           None if outer == 1 else "runtime_outer_stride"),
         *(OperandDescriptor(slot, OperandAccessKind.SCALAR, 0,
                             None if outer == 1 else 0, "SCALAR")
           for slot in range(2, length + 1))),
        OutputDescriptor(OperandAccessKind.CONTIGUOUS, "element_size",
                         None if outer == 1 else "runtime_output_outer_stride",
                         "FRESH_PROVIDER_OUTPUT"))
    ir = replace(original.elementwise_ir, layout_class="affine")
    candidate = ElementwiseOptimizer().legal_candidates(ir, iteration_descriptor=descriptor)[0]
    artifact = CBackend().generate(candidate)
    assert json.loads(artifact.manifest)["entry_abi"] == "affine_v2"
    artifact.write(tmp_path)
    library_path = tmp_path / "affine.so"
    subprocess.run(["cc", "-std=c11", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                    str(tmp_path / artifact.source_name), "-o", str(library_path)], check=True)
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, artifact.symbol)
    ptrdiff = ctypes.c_ssize_t
    function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ptrdiff),
                         ctypes.POINTER(ptrdiff), ctypes.POINTER(ctypes.c_double),
                         ctypes.c_size_t, ctypes.c_size_t]
    function.restype = None
    inner = 7
    a = (ctypes.c_double * (outer * inner))(*(i + .125 for i in range(outer * inner)))
    b = (ctypes.c_double * (outer * inner * 3))(*(i + .25 for i in range(outer * inner * 3)))
    scalar = ctypes.c_double(.375)
    out = (ctypes.c_double * (outer * inner))()
    # Reuse one artifact for arbitrary observed step sizes and both signs.
    for step in (1, 2, 3):
        offset = ((outer - 1) * inner * 3 + (inner - 1) * step) if reverse else 0
        inputs = (ctypes.c_void_p * (length + 1))(ctypes.addressof(a),
                    ctypes.addressof(b) + offset * 8,
                    *([ctypes.addressof(scalar)] * (length - 1)))
        inner_strides = (ptrdiff * (length + 1))(
            8, -step * 8 if reverse else step * 8, *([0] * (length - 1)))
        outer_strides = (ptrdiff * (length + 1))(inner * 8,
                            -inner * 3 * 8 if reverse else inner * 3 * 8,
                            *([0] * (length - 1)))
        function(inputs, inner_strides, outer_strides, out, outer, inner)
        left = [a[row * inner + i] for row in range(outer) for i in range(inner)]
        right = [b[offset + (-row * inner * 3 - i * step if reverse
                            else row * inner * 3 + i * step)]
                 for row in range(outer) for i in range(inner)]
        expected = _reference([left, right] +
                              [[scalar.value] * len(left)] * (length - 1),
                              length, False)
        assert list(out) == expected


def _recipes():
    recipes = {}
    for name in ("add", "subtract", "multiply", "true_divide"):
        recipe = Recipe(
            operation_id=f"provider.{name}", arity=2,
            semantic_expression=SemanticExpr.operation(
                name, SemanticExpr.argument(0), SemanticExpr.argument(1)),
            iteration_kind=IterationKind.ELEMENTWISE,
            effects=EffectSchema(
                EffectKind.PURE, MutationKind.NONE, AliasKind.FRESH, False),
            errors=ErrorSchema(
                True, FloatingPointErrorKind.PROVIDER_DEFINED),
        )
        recipes[recipe.operation_id] = recipe
    return recipes


def _candidate(length=4, branched=False, *, inner_contiguous=False):
    if branched:
        inputs = ("a", "b", "c", "d")
        operations = (
            HostOperation("n0", "provider.multiply", ("a", "b"), "t0"),
            HostOperation("n1", "provider.multiply", ("c", "d"), "t1"),
            HostOperation("n2", "provider.add", ("t0", "t1"), "out"),
        )
        virtual = ("t0", "t1")
        proof = "proof-branched"
    else:
        names = ("add", "multiply", "subtract", "true_divide", "add", "multiply")
        inputs = tuple(chr(ord("a") + index) for index in range(length + 1))
        operations_list = []
        prior = None
        for index in range(length):
            name = names[index % len(names)]
            result = "out" if index == length - 1 else f"t{index}"
            operations_list.append(HostOperation(
                f"n{index}", f"provider.{name}",
                (inputs[0] if index == 0 else prior, inputs[index + 1]),
                result))
            prior = result
        operations = tuple(operations_list)
        virtual = tuple(operation.result for operation in operations[:-1])
        proof = f"proof-{length}"
    region = HostProvenRegion(
        provider="provider", semantic_proof_digest=proof, inputs=inputs,
        operations=operations, outputs=("out",), virtual_values=virtual,
        materialized_values=("out",))
    specialization = ProviderSpecialization(
        contract_digest="contract", dtype_class="fp64",
        layout_class=("inner_contiguous" if inner_contiguous else
                      "contiguous_same_shape"),
        error_mode="provider_fpe_fallback")
    ir = build_elementwise_ir(region, _recipes(), specialization)
    descriptor = None
    if inner_contiguous:
        descriptor = IterationDescriptor(
            1, IterationRankClass.OUTER_INNER, "runtime_outer_count",
            "AFFINE_ROWS", "runtime_inner_extent",
            tuple(OperandDescriptor(
                index, OperandAccessKind.CONTIGUOUS, "element_size",
                "runtime_outer_stride") for index in range(len(inputs))),
            OutputDescriptor(
                OperandAccessKind.CONTIGUOUS, "element_size",
                "runtime_output_outer_stride", "FRESH_PROVIDER_OUTPUT"),
        )
    return ElementwiseOptimizer().legal_candidates(
        ir, iteration_descriptor=descriptor)[0]


def _reference(inputs, length, branched):
    if branched:
        return [left * right + third * fourth
                for left, right, third, fourth in zip(*inputs, strict=True)]
    output = []
    for values in zip(*inputs, strict=True):
        result = values[0]
        for index in range(length):
            operand = values[index + 1]
            kind = index % 6
            if kind in (0, 4):
                result = result + operand
            elif kind in (1, 5):
                result = result * operand
            elif kind == 2:
                result = result - operand
            else:
                result = result / operand
        output.append(result)
    return output


def _compile_and_run(tmp_path: Path, length: int, branched: bool):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    artifacts = CBackend().generate(_candidate(length, branched))
    header_path, source_path, _ = artifacts.write(tmp_path)
    library_path = tmp_path / "generated.so"
    subprocess.run([
        compiler, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
        "-ffp-contract=off", "-fPIC", "-shared", str(source_path),
        "-I", str(header_path.parent), "-o", str(library_path),
    ], check=True)
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, artifacts.symbol)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    size = 257
    input_count = 4 if branched else length + 1
    inputs = [array("d", (1.0 + index * 0.125 + i / size
                          for i in range(size)))
              for index in range(input_count)]
    buffers = [(ctypes.c_double * size).from_buffer(values)
               for values in inputs]
    pointers = (ctypes.c_void_p * input_count)(
        *(ctypes.cast(value, ctypes.c_void_p).value for value in buffers))
    output = (ctypes.c_double * size)()
    function(pointers, output, size)
    assert list(output) == pytest.approx(
        _reference(inputs, length, branched), rel=1e-15, abs=0.0)
    return artifacts


@pytest.mark.parametrize("length", (1, 2, 3, 4, 5, 6, 8, 12))
def test_generated_linear_kernels_compile_and_execute(tmp_path, length):
    _compile_and_run(tmp_path, length, False)


def test_generated_branched_kernel_compiles_and_executes(tmp_path):
    _compile_and_run(tmp_path, 3, True)


def test_generated_artifacts_are_deterministic_and_dispatch_free():
    candidate = _candidate(6)
    first = CBackend().generate(candidate)
    second = CBackend().generate(candidate)
    assert first == second
    assert first.symbol.startswith("crossjit_elem_fp64_contig_")
    assert first.source.count("for (") == 1
    for forbidden in ("switch", "opcode", "PyObject", "Python.h", "malloc"):
        assert forbidden not in first.source
    expected = (" + ", " * ", " - ", " / ", " + ", " * ")
    positions = []
    start = 0
    for operator in expected:
        position = first.source.index(operator, start)
        positions.append(position)
        start = position + len(operator)
    assert positions == sorted(positions)


def test_bundle_is_deterministic_and_compiles(tmp_path):
    candidates = tuple(_candidate(length) for length in range(2, 7))
    first = generate_bundle(candidates)
    second = generate_bundle(tuple(reversed(candidates)))
    assert first == second
    assert first.source.count("for (") == 5
    assert first.header.count("crossjit_elem_fp64_contig_") == 5
    assert '"kernels": [' in first.metadata
    source = tmp_path / "generated_crossjit_kernels.c"
    header = tmp_path / "generated_crossjit_kernels.h"
    source.write_text(first.source)
    header.write_text(first.header)
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    subprocess.run([
        compiler, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
        "-ffp-contract=off", "-c", str(source), "-I", str(tmp_path),
        "-o", str(tmp_path / "generated.o"),
    ], check=True)


def test_empty_bundle_is_valid_for_conservative_stale_build():
    bundle = generate_bundle(())
    assert json.loads(bundle.metadata)["kernels"] == []
    assert "Generated by crossjit-generator" in bundle.source


@pytest.mark.parametrize("length", (1, 2, 3, 4, 5, 6, 8, 12))
@pytest.mark.parametrize("outer,inner", ((1, 17), (3, 31), (7, 8)))
def test_inner_contiguous_linear_kernels_compile_and_execute(
        tmp_path, length, outer, inner):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    candidate = _candidate(length, inner_contiguous=True)
    artifacts = CBackend().generate(candidate)
    header_path, source_path, _ = artifacts.write(tmp_path)
    library_path = tmp_path / "generated_inner.so"
    subprocess.run([
        compiler, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
        "-ffp-contract=off", "-fPIC", "-shared", str(source_path),
        "-I", str(header_path.parent), "-o", str(library_path),
    ], check=True)
    function = getattr(ctypes.CDLL(str(library_path)), artifacts.symbol)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_ssize_t),
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t,
    ]
    input_count = length + 1
    padded = inner + 3
    inputs = [array("d", (1.0 + slot * 0.125 + i / (outer * padded)
                          for i in range(outer * padded)))
              for slot in range(input_count)]
    buffers = [(ctypes.c_double * len(value)).from_buffer(value)
               for value in inputs]
    pointers = (ctypes.c_void_p * input_count)(
        *(ctypes.cast(value, ctypes.c_void_p).value for value in buffers))
    strides = (ctypes.c_ssize_t * input_count)(
        *((padded * ctypes.sizeof(ctypes.c_double),) * input_count))
    output = (ctypes.c_double * (outer * inner))()
    function(pointers, strides, output, outer, inner)
    expected = []
    for row in range(outer):
        row_inputs = [value[row * padded:row * padded + inner]
                      for value in inputs]
        expected.extend(_reference(row_inputs, length, False))
    assert list(output) == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert artifacts.source.count("for (") == 2
    inner_loop = artifacts.source.split(
        "for (size_t i = 0; i < inner_extent; i++)", 1)[1]
    for forbidden in ("switch", "opcode", "PyObject", "Python.h"):
        assert forbidden not in inner_loop


def test_inner_contiguous_branched_kernel_is_structural():
    candidate = _candidate(3, branched=True, inner_contiguous=True)
    artifacts = CBackend().generate(candidate)
    assert candidate.lowering_form == "outer_inner_compiler_visible_loop"
    assert artifacts.symbol.startswith("crossjit_elem_fp64_inner_contig_")
    assert json.loads(artifacts.manifest)["entry_abi"] == "outer_inner_v1"
    assert "outer_strides" in artifacts.source
    assert artifacts.source.index(" * ") < artifacts.source.rindex(" + ")


def test_outer_inner_scalar_and_zero_stride_compile_and_execute(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    base = _candidate(2)
    descriptor = IterationDescriptor(
        1, IterationRankClass.OUTER_INNER, "runtime_outer_count",
        "AFFINE_ROWS", "runtime_inner_extent",
        (
            OperandDescriptor(0, OperandAccessKind.CONTIGUOUS,
                              "element_size", "runtime_outer_stride"),
            OperandDescriptor(1, OperandAccessKind.SCALAR, 0, 0, "SCALAR"),
            OperandDescriptor(2, OperandAccessKind.ZERO_STRIDE, 0,
                              "runtime_outer_stride", "BROADCAST_INNER"),
        ),
        OutputDescriptor(
            OperandAccessKind.CONTIGUOUS, "element_size",
            "runtime_output_outer_stride", "FRESH_PROVIDER_OUTPUT"),
    )
    candidate = ElementwiseOptimizer().legal_candidates(
        base.elementwise_ir, iteration_descriptor=descriptor)[0]
    artifacts = CBackend().generate(candidate)
    header_path, source_path, _ = artifacts.write(tmp_path)
    library_path = tmp_path / "generated_scalar_zero.so"
    subprocess.run([
        compiler, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
        "-ffp-contract=off", "-fPIC", "-shared", str(source_path),
        "-I", str(header_path.parent), "-o", str(library_path),
    ], check=True)
    function = getattr(ctypes.CDLL(str(library_path)), artifacts.symbol)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_ssize_t),
        ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t,
    ]
    outer, inner, padded = 3, 7, 9
    contiguous = array("d", (1 + i / 10 for i in range(outer * padded)))
    scalar = array("d", (0.25,))
    per_row = array("d", (2.0, 3.0, 4.0))
    buffers = [
        (ctypes.c_double * len(values)).from_buffer(values)
        for values in (contiguous, scalar, per_row)
    ]
    pointers = (ctypes.c_void_p * 3)(
        *(ctypes.cast(value, ctypes.c_void_p).value for value in buffers))
    strides = (ctypes.c_ssize_t * 3)(
        padded * ctypes.sizeof(ctypes.c_double), 0,
        ctypes.sizeof(ctypes.c_double))
    output = (ctypes.c_double * (outer * inner))()
    function(pointers, strides, output, outer, inner)
    expected = [
        (contiguous[row * padded + column] + scalar[0]) * per_row[row]
        for row in range(outer) for column in range(inner)
    ]
    assert list(output) == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert "input_1[0]" in artifacts.source
    assert "input_2[0]" in artifacts.source
    assert artifacts.source.count("for (") == 2


def test_iteration_layout_specializes_execution_not_structural_signature():
    candidate = _candidate(2, inner_contiguous=True)
    signature = optimization_signature_from_ir(
        candidate.elementwise_ir, provider_abi_version=3)
    specialization = candidate.execution_specialization(
        signature.stable_hash())

    changed_descriptor = replace(
        candidate.iteration_descriptor,
        operands=tuple(
            replace(item, outer_stride="alternate_runtime_outer_stride")
            for item in candidate.iteration_descriptor.operands),
    )
    changed_candidate = replace(
        candidate, iteration_descriptor=changed_descriptor)
    changed_specialization = changed_candidate.execution_specialization(
        signature.stable_hash())

    assert signature == optimization_signature_from_ir(
        changed_candidate.elementwise_ir, provider_abi_version=3)
    assert specialization.structural_signature_hash == signature.stable_hash()
    assert (changed_specialization.structural_signature_hash ==
            signature.stable_hash())
    assert specialization.stable_hash() != changed_specialization.stable_hash()
