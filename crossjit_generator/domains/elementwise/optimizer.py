"""Conservative first ElementwiseIR optimizer.

This optimizer performs execution synthesis only.  It does not reassociate,
rewrite, or establish Host legality, and it does not select candidates for
production profitability.
"""

from __future__ import annotations

from crossjit_generator.candidate import Candidate, CostHints, RuntimeAssumption
from crossjit_generator.ir import ElementwiseIR
from crossjit_generator.iteration import (
    IterationDescriptor,
    IterationRankClass,
)


class UnsupportedElementwiseIR(ValueError):
    """The supplied, Host-proven IR is outside this optimizer's domain."""


class ElementwiseOptimizer:
    VERSION = "1"
    _DTYPES = frozenset(("fp64", "float64"))
    _LAYOUTS = frozenset(("contiguous_same_shape", "inner_contiguous"))
    _OPERATORS = frozenset(("add", "subtract", "multiply", "true_divide"))

    def legal_candidates(
        self, ir: ElementwiseIR, *,
        iteration_descriptor: IterationDescriptor | None = None,
    ) -> tuple[Candidate, ...]:
        if not isinstance(ir, ElementwiseIR):
            raise UnsupportedElementwiseIR("expected ElementwiseIR")
        if ir.dtype_class not in self._DTYPES:
            raise UnsupportedElementwiseIR(
                f"unsupported dtype class: {ir.dtype_class}")
        if ir.layout_class not in self._LAYOUTS:
            raise UnsupportedElementwiseIR(
                f"unsupported layout class: {ir.layout_class}")
        if (iteration_descriptor is not None and
                len(iteration_descriptor.operands) != len(ir.inputs)):
            raise UnsupportedElementwiseIR(
                "iteration operand count does not match IR inputs")
        if ir.layout_class == "inner_contiguous" and (
                iteration_descriptor is None or
                iteration_descriptor.iteration_rank_class is not
                IterationRankClass.OUTER_INNER):
            raise UnsupportedElementwiseIR(
                "inner-contiguous IR requires outer/inner descriptor")
        # Historical structural layout names are intentionally not used to
        # select loop rank.  The Provider-owned descriptor is the execution
        # authority, including for scalar/broadcast specializations whose
        # structural signature remains ``contiguous_same_shape``.
        if not 2 <= len(ir.operations) <= 6:
            raise UnsupportedElementwiseIR(
                "MVP requires two through six operations")
        if len(ir.outputs) != 1 or ir.outputs != ir.materialized_values:
            raise UnsupportedElementwiseIR(
                "MVP requires one final materialized output")
        known = set(ir.inputs)
        signature = []
        for operation in ir.operations:
            if any(operand not in known for operand in operation.operands):
                raise UnsupportedElementwiseIR(
                    "operation dependencies are not topologically ordered")
            expression = operation.semantic_expression
            operator = expression.get("operator")
            if operator not in self._OPERATORS:
                raise UnsupportedElementwiseIR(
                    f"unsupported semantic operator: {operator}")
            effects = operation.effects
            if (effects.get("effect") != "pure" or
                    effects.get("mutation") != "none" or
                    effects.get("result_alias") != "fresh" or
                    effects.get("may_call_host") is not False):
                raise UnsupportedElementwiseIR(
                    "operation effect schema is not supported")
            if operation.errors.get("preserve_operation_order") is not True:
                raise UnsupportedElementwiseIR(
                    "operation-order relaxation is not supported")
            known.add(operation.result)
            signature.append(operation.operation_id)
        assumptions = [
            RuntimeAssumption("dtype_class", ir.dtype_class),
            RuntimeAssumption("error_mode", ir.error_mode),
            RuntimeAssumption("layout_class", ir.layout_class),
        ]
        assumptions.extend(
            RuntimeAssumption(attribute.name, attribute.value)
            for attribute in ir.specialization_attributes
        )
        assumptions.sort(key=lambda item: item.name)
        lowering_form = (
            "outer_inner_compiler_visible_loop"
            if iteration_descriptor is not None and
            iteration_descriptor.iteration_rank_class is
            IterationRankClass.OUTER_INNER
            else "single_compiler_visible_loop"
        )
        candidate = Candidate(
            schema_version=1,
            domain="elementwise",
            elementwise_ir=ir,
            operation_signature=tuple(signature),
            lowering_form=lowering_form,
            runtime_assumptions=tuple(assumptions),
            target_requirements=("c11", "strict_operation_order"),
            cost_hints=CostHints(
                operation_count=len(ir.operations),
                input_count=len(ir.inputs),
                virtual_intermediate_count=len(ir.virtual_intermediates),
            ),
            iteration_descriptor=iteration_descriptor,
        )
        return (candidate,)
