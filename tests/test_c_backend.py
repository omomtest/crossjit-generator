from __future__ import annotations

from array import array
import ctypes
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


def _candidate(length=4, branched=False):
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
        for index, name in enumerate(names[:length]):
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
        layout_class="contiguous_same_shape",
        error_mode="provider_fpe_fallback")
    ir = build_elementwise_ir(region, _recipes(), specialization)
    return ElementwiseOptimizer().legal_candidates(ir)[0]


def _reference(inputs, length, branched):
    if branched:
        return [left * right + third * fourth
                for left, right, third, fourth in zip(*inputs, strict=True)]
    output = []
    for values in zip(*inputs, strict=True):
        result = values[0] + values[1]
        if length >= 2:
            result *= values[2]
        if length >= 3:
            result -= values[3]
        if length >= 4:
            result /= values[4]
        if length >= 5:
            result += values[5]
        if length >= 6:
            result *= values[6]
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


@pytest.mark.parametrize("length", range(2, 7))
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
