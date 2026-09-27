# CrossJIT Generator G6 oracle parity — 2026-09-27

Decision:

```text
G6 = PASS
```

The C backend reproduces the handwritten G0 oracle for the exact L2/L4/L6
signatures across the complete 13-point G0/P1 N grid. NumPy build/runtime
integration may proceed.

## Method

For each signature the gate compiled:

```text
handwritten G0 oracle translation unit
generated CBackend translation unit
a shared parity harness
```

with the same GCC 13.3 compiler and flags:

```text
-std=c11 -O3 -fno-strict-aliasing -ffp-contract=off
-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2
-mpopcnt -msahf -mcx16
```

No fast-math option was used. The harness uses identical aligned inputs and
preallocated outputs, alternates oracle/generated measurement order, and runs
five independent processes with seven samples per cell.

Measured cells:

```text
3 signatures × 13 N values = 39 cells
N = 16 through 65536 on the complete G0 grid
```

## Correctness

All 39 cells pass:

```text
finite bit/value parity
NaN parity
Inf parity
signed-zero parity
FE_DIVBYZERO parity
FE_INVALID parity
FE_OVERFLOW parity
FE_UNDERFLOW parity
```

The operation sequence remains exactly:

```text
L2: add, multiply
L4: add, multiply, subtract, true_divide
L6: add, multiply, subtract, true_divide, add, multiply
```

## Performance parity

Representative medians are nanoseconds per loop invocation:

| case | N | handwritten oracle | generated C | generated/oracle |
| --- | ---: | ---: | ---: | ---: |
| L2 | 256 | 41.60 | 37.91 | 0.911x |
| L4 | 4096 | 1547.18 | 1547.13 | 1.000x |
| L6 | 65536 | 48520.41 | 48529.76 | 1.000x |

Median generated/oracle ratio across all 13 N values for each signature:

```text
L2: 1.0110x
L4: 0.9999x
L6: 1.0002x
```

The G6 performance envelope was `generated/oracle <= 1.15` at each required
representative point and for the median across each complete signature grid.
All checks pass. Tiny-N cells remain dominated by function/timer noise; they
do not indicate a generated-code shape difference.

## Compiler/code-shape evidence

GCC reports every generated loop vectorized with 16-byte vectors.

After removing alignment padding NOPs, `objdump` instruction-mnemonic shapes
are identical between handwritten and generated functions for L2, L4, and L6.
The ordered packed arithmetic is the same, including `divpd` for L4/L6.

Generated source contains:

```text
one loop
straight-line scalar temporaries
one final output store
```

and contains no:

```text
opcode switch
semantic-node traversal
generic slot interpreter
Python dispatch
runtime compiler hook
```

## Evidence

```text
tools/run_oracle_parity.py
results/g6_oracle_parity.json
results/g6_oracle_parity.csv
g6-build/*/vectorization.txt
g6-build/g0_oracle_vectorization.txt
g6-build/g6_disassembly.txt
```

`g6-build/` is reproducible and intentionally ignored; the source controller
and machine-readable aggregate results are versioned.

## Gate conclusion

Generated C is not materially slower than the handwritten feasibility oracle.
It has the same strict operation order, tested observable FP behavior,
vectorization status, and normalized assembly shape. Therefore:

```text
G6 = PASS
Next: G7 NumPy build-time generation integration.
```
