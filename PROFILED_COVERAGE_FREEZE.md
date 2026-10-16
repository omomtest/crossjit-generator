# GEN-PROFILED Generator schema freeze

> Status: Frozen implementation report / non-normative.
>
> The only normative CrossJIT v3 specification is defined by
> [crossjit_v3.md](../crossjit_v3.md), [crossjit_v3_plan.md](../crossjit_v3_plan.md), [crossjit_v3_goal.md](../crossjit_v3_goal.md).
> If this document conflicts with those files, the normative specification takes precedence.
> Freeze restrictions apply to the recorded baseline. Future evidence-gated configurations follow the normative roadmap and receive separate checkpoints without moving frozen tags.

Date: 2026-09-30

PF6 remeasured the final NumPy PROFILED candidate universe without adding a
new execution class. The same 24 structural signatures and 32 refined profile
variants were observed; the bounded build emits 28 kernels, including 17
profiled execution specializations. Eight execution specializations are
selected and two measured specializations are rejected. Missing, stale,
unmeasured, and unprofitable specializations fail closed.

The fixed-affine residual is explicitly deferred and truly general striding
is unsupported. Neither frequency nor artifact existence authorizes a new
runtime specialization without independent generic-signature measurements.

```text
GENERATOR_SCHEMA_FREEZE
```

Only correctness bugs may change these interfaces and formats after this
checkpoint:

```text
Recipe schema
ElementwiseIR schema
ExecutionSpecialization schema
ElementwiseOptimizer interface
C backend interface
runtime registry artifact format
OptimizationSignature format
DecisionDB and ExecutionDecisionDB formats
```

The freeze preserves the abstraction boundary:

```text
OptimizationSignature = structural computation identity
ExecutionSpecialization = Provider-owned execution/layout identity
DecisionKey = structural + execution specialization + environment
```

Schema/interface source hashes:

```text
recipe/model.py:                       6cad8f8b971a550d3df5bfb94f6c3aa2b9fe79527e521fd87860edbce38f90be
ir/elementwise.py:                     d2877fd0714e82f0727889c13b08dc04b6e0670ae5e0f3bf94e79e4f24464c4b
candidate/model.py:                    8d3aa3e7f91ad697393fcecc34e139eb354e72edf264ba2395031adad14e36d5
domains/elementwise/optimizer.py:       6b13452442374bbae9096d14f303f6758ee169fe71b1460384454ebe6b49b552
backend/c/backend.py:                  78fa2485412a2f714abf15e13f23895af4796096857440530a50b1755f9bfaee
signature/model.py:                    387d039471b72b48516c23d8e8d425a52186e41a0870aa88b2dceee21f664b12
decision/model.py:                     b16886b393e21534de3fd8a5c084116f0d4013562b880bef09143f1ae617cb12
ordered source-hash-set SHA-256:        a7602c182cabef94135124f4118a6651abcc9d80feb1e81eb6e7b14958410a6f
```

Validation at the freeze candidate: 87 tests passed. No formal NPBench timing
was performed.

