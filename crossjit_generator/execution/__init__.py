"""Bounded execution synthesis; descriptors contain no extension objects."""
from .model import ExecutionLimits, Primitive, ScalarLoop, ScalarType
from .c import generate_backend_header, generate_execution_header, generate_loop_header

__all__ = ['ExecutionLimits', 'Primitive', 'ScalarLoop', 'ScalarType',
           'generate_backend_header', 'generate_execution_header', 'generate_loop_header']
