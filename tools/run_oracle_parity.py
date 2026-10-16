#!/usr/bin/env python3
"""G6 generated-C parity gate against the handwritten G0 oracle."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from crossjit_generator.backend.c import CBackend
from crossjit_generator.domains.elementwise import ElementwiseOptimizer


CASES = {
    "linear_2": ("add", "mul"),
    "linear_4": ("add", "mul", "sub", "true_divide"),
    "linear_6": ("add", "mul", "sub", "true_divide", "add", "mul"),
}
DEFAULT_N = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
             16384, 32768, 65536)
REPRESENTATIVE = {"linear_2": 256, "linear_4": 4096, "linear_6": 65536}
FLAGS = (
    "-std=c11", "-O3", "-fno-strict-aliasing", "-ffp-contract=off",
    "-msse", "-msse2", "-msse3", "-mssse3", "-msse4.1", "-msse4.2",
    "-mpopcnt", "-msahf", "-mcx16",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(case_id: str, operations: tuple[str, ...]):
    length = len(operations)
    inputs = [chr(ord("a") + index) for index in range(length + 1)]
    nodes = []
    prior = None
    for index, operation in enumerate(operations):
        result = "out" if index == length - 1 else f"t{index}"
        nodes.append({
            "id": f"n{index}",
            "operation": operation,
            "operands": [inputs[0] if index == 0 else prior, inputs[index + 1]],
            "result": result,
        })
        prior = result
    return {
        "provider": "numpy",
        "host_semantic_proof_digest": f"g0-proof-{case_id}",
        "inputs": inputs,
        "operations": nodes,
        "outputs": ["out"],
        "virtual_values": [node["result"] for node in nodes[:-1]],
        "materialized_values": ["out"],
        "specialization": {
            "contract_digest": "g0-fp64-contiguous-contract",
            "dtype_class": "float64",
            "layout_class": "contiguous_same_shape",
            "error_mode": "numpy_fpe_fallback",
        },
    }


def _harness_source(symbols: dict[str, str]) -> str:
    declarations = []
    entries = []
    for case_id, operations in CASES.items():
        length = len(operations)
        suffix = case_id.removeprefix("linear_")
        symbol = symbols[case_id]
        declarations.extend((
            f"void g0_oracle_linear_{suffix}(const double *const *, "
            "double *restrict, size_t);",
            f"void {symbol}(const void *const *, double *restrict, size_t);",
            f"static void oracle_{suffix}(const void *const *inputs, "
            "double *restrict output, size_t size)",
            "{",
            f"    g0_oracle_linear_{suffix}((const double *const *)inputs, "
            "output, size);",
            "}",
        ))
        entries.append(
            f'    {{"{case_id}", {length}, {length + 1}, oracle_{suffix}, '
            f"{symbol}}},")
    return "\n".join((
        "#define _POSIX_C_SOURCE 200809L",
        "#include <fenv.h>",
        "#include <math.h>",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <time.h>",
        "",
        "typedef void (*kernel_fn)(const void *const *, double *restrict, size_t);",
        *declarations,
        "",
        "typedef struct {",
        "    const char *name;",
        "    int length;",
        "    size_t input_count;",
        "    kernel_fn oracle;",
        "    kernel_fn generated;",
        "} Case;",
        "",
        "static const Case cases[] = {",
        *entries,
        "};",
        "",
        "static uint64_t bits(double value)",
        "{",
        "    uint64_t result;",
        "    memcpy(&result, &value, sizeof(result));",
        "    return result;",
        "}",
        "",
        "static int same_value(double left, double right)",
        "{",
        "    return (isnan(left) && isnan(right)) || bits(left) == bits(right);",
        "}",
        "",
        "static uint64_t clock_ns(void)",
        "{",
        "    struct timespec now;",
        "    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) exit(2);",
        "    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;",
        "}",
        "",
        "static double median(double *values, int count)",
        "{",
        "    for (int i = 1; i < count; i++) {",
        "        double value = values[i];",
        "        int j = i;",
        "        while (j > 0 && values[j - 1] > value) {",
        "            values[j] = values[j - 1];",
        "            j--;",
        "        }",
        "        values[j] = value;",
        "    }",
        "    return values[count / 2];",
        "}",
        "",
        "static int validate_pair(const Case *item, const void *const *inputs,",
        "                         double *oracle, double *generated, size_t size)",
        "{",
        "    feclearexcept(FE_ALL_EXCEPT);",
        "    item->oracle(inputs, oracle, size);",
        "    int oracle_fpe = fetestexcept(FE_DIVBYZERO | FE_INVALID |",
        "                                  FE_OVERFLOW | FE_UNDERFLOW);",
        "    feclearexcept(FE_ALL_EXCEPT);",
        "    item->generated(inputs, generated, size);",
        "    int generated_fpe = fetestexcept(FE_DIVBYZERO | FE_INVALID |",
        "                                     FE_OVERFLOW | FE_UNDERFLOW);",
        "    if (oracle_fpe != generated_fpe) return 0;",
        "    for (size_t i = 0; i < size; i++)",
        "        if (!same_value(oracle[i], generated[i])) return 0;",
        "    return 1;",
        "}",
        "",
        "static double run(kernel_fn function, const void *const *inputs,",
        "                  double *output, size_t size, size_t iterations)",
        "{",
        "    uint64_t started = clock_ns();",
        "    for (size_t i = 0; i < iterations; i++)",
        "        function(inputs, output, size);",
        "    return (double)(clock_ns() - started) / (double)iterations;",
        "}",
        "",
        "int main(int argc, char **argv)",
        "{",
        "    if (argc != 6) return 2;",
        "    const Case *item = NULL;",
        "    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)",
        "        if (strcmp(argv[1], cases[i].name) == 0) item = &cases[i];",
        "    if (item == NULL) return 2;",
        "    size_t size = strtoull(argv[2], NULL, 10);",
        "    int samples = atoi(argv[3]);",
        "    size_t target = strtoull(argv[4], NULL, 10);",
        "    size_t cap = strtoull(argv[5], NULL, 10);",
        "    if (samples < 1 || samples > 31 || size < 8) return 2;",
        "    double *storage[7] = {0};",
        "    const void *inputs[7] = {0};",
        "    for (size_t input = 0; input < item->input_count; input++) {",
        "        if (posix_memalign((void **)&storage[input], 64,",
        "                           size * sizeof(double)) != 0) return 2;",
        "        inputs[input] = storage[input];",
        "        for (size_t i = 0; i < size; i++)",
        "            storage[input][i] = 1.0 + (double)i / (double)size +",
        "                                (double)input * 0.125;",
        "    }",
        "    double *oracle = NULL;",
        "    double *generated = NULL;",
        "    if (posix_memalign((void **)&oracle, 64, size * sizeof(double)) != 0 ||",
        "        posix_memalign((void **)&generated, 64, size * sizeof(double)) != 0)",
        "        return 2;",
        "    if (!validate_pair(item, inputs, oracle, generated, size)) return 3;",
        "    static const double special[] =",
        "        {NAN, INFINITY, -INFINITY, -0.0, 0.0, 1.0, -1.0, 2.0};",
        "    for (size_t input = 0; input < item->input_count; input++)",
        "        for (size_t i = 0; i < 8; i++)",
        "            storage[input][i] = special[(i + input) % 8];",
        "    if (!validate_pair(item, inputs, oracle, generated, 8)) return 3;",
        "    for (size_t input = 0; input < item->input_count; input++)",
        "        for (size_t i = 0; i < size; i++)",
        "            storage[input][i] = 1.0 + (double)i / (double)size +",
        "                                (double)input * 0.125;",
        "    size_t iterations = target / size;",
        "    if (iterations < 8) iterations = 8;",
        "    if (iterations > cap) iterations = cap;",
        "    for (int i = 0; i < 32; i++) {",
        "        item->oracle(inputs, oracle, size);",
        "        item->generated(inputs, generated, size);",
        "    }",
        "    double oracle_samples[31];",
        "    double generated_samples[31];",
        "    for (int sample = 0; sample < samples; sample++) {",
        "        if (sample & 1) {",
        "            generated_samples[sample] = run(item->generated, inputs, generated, size, iterations);",
        "            oracle_samples[sample] = run(item->oracle, inputs, oracle, size, iterations);",
        "        } else {",
        "            oracle_samples[sample] = run(item->oracle, inputs, oracle, size, iterations);",
        "            generated_samples[sample] = run(item->generated, inputs, generated, size, iterations);",
        "        }",
        "    }",
        "    double oracle_median = median(oracle_samples, samples);",
        "    double generated_median = median(generated_samples, samples);",
        '    printf("{\\"case_identifier\\":\\"%s\\",\\"element_count\\":%zu,",',
        "           item->name, size);",
        '    printf("\\"oracle_median_ns\\":%.17g,\\"generated_median_ns\\":%.17g,",',
        "           oracle_median, generated_median);",
        "    printf(\"\\\"correct\\\":true,\\\"fpe_parity\\\":true}\\n\");",
        "    return 0;",
        "}",
        "",
    ))


def _compile(args):
    args.build_dir.mkdir(parents=True, exist_ok=True)
    lowering = _load_module(
        "numpy_crossjit_lowering",
        args.numpy_root / "numpy/_core/code_generators/crossjit_lowering.py")
    symbols = {}
    generated_sources = {}
    objects = []
    diagnostics = {}
    for case_id, operations in CASES.items():
        ir = lowering.lower_manifest(_manifest(case_id, operations))
        candidate = ElementwiseOptimizer().legal_candidates(ir)[0]
        artifacts = CBackend().generate(candidate)
        case_dir = args.build_dir / case_id
        _, source_path, _ = artifacts.write(case_dir)
        symbols[case_id] = artifacts.symbol
        generated_sources[case_id] = artifacts.source
        object_path = case_dir / "generated.o"
        diagnostic_path = case_dir / "vectorization.txt"
        subprocess.run([
            args.cc, *FLAGS, f"-fopt-info-vec-all={diagnostic_path}",
            "-I", str(case_dir), "-c", str(source_path), "-o", str(object_path),
        ], check=True)
        objects.append(object_path)
        diagnostics[case_id] = diagnostic_path

    oracle_object = args.build_dir / "g0_oracle.o"
    oracle_diagnostics = args.build_dir / "g0_oracle_vectorization.txt"
    subprocess.run([
        args.cc, *FLAGS, "-DG0_ORACLE_LIBRARY_ONLY",
        f"-fopt-info-vec-all={oracle_diagnostics}", "-c", str(args.oracle_source),
        "-o", str(oracle_object),
    ], check=True)
    harness_path = args.build_dir / "g6_parity_harness.c"
    harness_path.write_text(_harness_source(symbols))
    harness_object = args.build_dir / "g6_parity_harness.o"
    subprocess.run([
        args.cc, *FLAGS, "-Wall", "-Wextra", "-Werror", "-c",
        str(harness_path), "-o", str(harness_object),
    ], check=True)
    executable = args.build_dir / "g6_parity"
    subprocess.run([
        args.cc, str(harness_object), str(oracle_object),
        *(str(path) for path in objects), "-lm", "-o", str(executable),
    ], check=True)
    disassembly_path = args.build_dir / "g6_disassembly.txt"
    completed = subprocess.run(
        [args.objdump, "-d", str(executable)], check=True,
        capture_output=True, text=True)
    disassembly_path.write_text(completed.stdout)
    return executable, symbols, diagnostics, oracle_diagnostics, disassembly_path


def _measure(args, executable):
    jobs = [(process, case_id, n) for process in range(args.processes)
            for case_id in CASES for n in args.n_grid]
    random.Random(args.seed).shuffle(jobs)
    raw = []
    for process, case_id, n in jobs:
        completed = subprocess.run([
            str(executable), case_id, str(n), str(args.samples),
            str(args.target_elements), str(args.max_iterations),
        ], check=True, capture_output=True, text=True)
        row = json.loads(completed.stdout)
        row["independent_process_index"] = process
        raw.append(row)
    rows = []
    for case_id in CASES:
        for n in args.n_grid:
            records = [row for row in raw
                       if row["case_identifier"] == case_id and
                       row["element_count"] == n]
            oracle = [row["oracle_median_ns"] for row in records]
            generated = [row["generated_median_ns"] for row in records]
            ratios = [right / left for left, right in zip(oracle, generated,
                                                          strict=True)]
            rows.append({
                "case_identifier": case_id,
                "operation_sequence": ",".join(CASES[case_id]),
                "element_count": n,
                "dtype": "float64",
                "layout": "CONTIGUOUS_SAME_SHAPE",
                "oracle_median_ns": statistics.median(oracle),
                "generated_median_ns": statistics.median(generated),
                "generated_over_oracle": statistics.median(ratios),
                "generated_over_oracle_processes": ratios,
                "independent_processes": len(records),
                "correct": all(row["correct"] for row in records),
                "fpe_parity": all(row["fpe_parity"] for row in records),
            })
    return rows


def _instruction_shape(disassembly: str, symbol: str) -> tuple[str, ...]:
    marker = f"<{symbol}>:"
    active = False
    instructions = []
    for line in disassembly.splitlines():
        if line.rstrip().endswith(marker):
            active = True
            continue
        if active and line.rstrip().endswith(">:"):
            break
        if active and "\t" in line:
            fields = line.split("\t")
            if len(fields) >= 3:
                mnemonic = fields[2].split()[0]
                if mnemonic not in ("cs", "data16", "nop", "nopl"):
                    instructions.append(mnemonic)
    return tuple(instructions)


def main():
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--numpy-root", type=Path, default=workspace / "numpy")
    parser.add_argument("--oracle-source", type=Path, default=(
        workspace / "numpy/benchmarks/benchmarks/crossjit/g0_static_oracle.c"))
    parser.add_argument("--build-dir", type=Path, default=root / "g6-build")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-elements", type=int, default=100_000_000)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=620)
    parser.add_argument("--n-grid", type=lambda value: tuple(map(int, value.split(","))),
                        default=DEFAULT_N)
    args = parser.parse_args()

    (executable, symbols, diagnostics, oracle_diagnostics,
     disassembly) = _compile(args)
    rows = _measure(args, executable)
    vectorized = {}
    for case_id, path in diagnostics.items():
        vectorized[case_id] = "loop vectorized using 16 byte vectors" in path.read_text()
    representative = {
        case_id: next(row for row in rows
                      if row["case_identifier"] == case_id and
                      row["element_count"] == n)
        for case_id, n in REPRESENTATIVE.items()
    }
    within_envelope = {
        case_id: row["generated_over_oracle"] <= 1.15
        for case_id, row in representative.items()
    }
    grid_median_ratio = {
        case_id: statistics.median(
            row["generated_over_oracle"] for row in rows
            if row["case_identifier"] == case_id)
        for case_id in CASES
    }
    disassembly_text = disassembly.read_text()
    assembly_shape_equivalent = {
        case_id: _instruction_shape(
            disassembly_text, f"g0_oracle_{case_id}") ==
            _instruction_shape(disassembly_text, symbol)
        for case_id, symbol in symbols.items()
    }
    decision = (
        all(row["correct"] and row["fpe_parity"] for row in rows) and
        all(vectorized.values()) and
        all(within_envelope.values()) and
        all(ratio <= 1.15 for ratio in grid_median_ratio.values()) and
        all(assembly_shape_equivalent.values())
    )
    compiler_id = subprocess.run(
        [args.cc, "--version"], check=True, capture_output=True,
        text=True).stdout.splitlines()[0]
    document = {
        "schema_version": 1,
        "gate": "G6",
        "decision": "PASS" if decision else "FAIL",
        "configuration": {
            "compiler": compiler_id,
            "compiler_flags": list(FLAGS),
            "independent_processes": args.processes,
            "samples_per_process": args.samples,
            "target_elements": args.target_elements,
            "n_grid": list(args.n_grid),
            "performance_envelope_max_generated_over_oracle": 1.15,
            "unsafe_fast_math": False,
        },
        "compiler_evidence": {
            "generated_vectorized": vectorized,
            "assembly_shape_equivalent": assembly_shape_equivalent,
            "generated_diagnostics": {
                name: str(path) for name, path in diagnostics.items()
            },
            "oracle_diagnostics": str(oracle_diagnostics),
            "disassembly": str(disassembly),
            "symbols": symbols,
        },
        "representative": representative,
        "grid_median_generated_over_oracle": grid_median_ratio,
        "within_performance_envelope": within_envelope,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "g6_oracle_parity.json"
    csv_path = args.output_dir / "g6_oracle_parity.csv"
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(
            {key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "decision": document["decision"],
        "json": str(json_path),
        "csv": str(csv_path),
        "representative": representative,
        "vectorized": vectorized,
        "assembly_shape_equivalent": assembly_shape_equivalent,
        "grid_median_generated_over_oracle": grid_median_ratio,
    }, indent=2, sort_keys=True))
    return 0 if decision else 2


if __name__ == "__main__":
    raise SystemExit(main())
