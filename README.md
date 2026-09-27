# CrossJIT Generator

CrossJIT Generator is a build-time framework for synthesizing Provider-owned
execution implementations from Host-proven regions.  It does not form Host
regions, reinterpret Host proofs, or own Provider domain semantics.

The first reference pipeline is:

```text
Provider atomic recipes (Q_P)
+ runtime specialization contract (C_P)
+ Host region/proof (R_H, Pi_H)
        ↓
ElementwiseIR
        ↓
ElementwiseOptimizer
        ↓
Candidate
        ↓
C backend
        ↓
generated C/H compiled by the ordinary Provider build
```

Generation is offline/build-time only.  Runtime compilation and LLVM are out
of scope.  Provider-specific adapters and recipes, including NumPy recipes,
remain in their Provider repositories.

Run the tests from this repository with:

```bash
python -m pytest
```
