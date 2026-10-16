"""Build-time contracts for the bounded, precompiled execution engine.

Runtime values, stages, capabilities and synthesized lifetimes are emitted as
plain C descriptors. Providers validate semantic legality before projection.
"""
from dataclasses import dataclass
import re


def identifier(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value):
        raise ValueError(f'invalid C identifier: {value!r}')
    return value


@dataclass(frozen=True)
class ExecutionLimits:
    nodes: int = 12
    inputs: int = 16
    rank: int = 64

    def __post_init__(self):
        if not (1 <= self.nodes <= 255 and 1 <= self.inputs <= 255
                and self.nodes + self.inputs <= 32 and 1 <= self.rank <= 255):
            raise ValueError('execution bounds exceed compact descriptor capacity')


@dataclass(frozen=True)
class ScalarType:
    kind: str
    width: int

    def __post_init__(self):
        if (self.kind, self.width) not in {
                ('float', 32), ('float', 64), ('unsigned', 32), ('unsigned', 64)}:
            raise ValueError('unsupported scalar storage contract')

    @property
    def c_type(self):
        return ('float' if self.width == 32 else 'double') if self.kind == 'float' else f'uint{self.width}_t'


@dataclass(frozen=True)
class Primitive:
    """Opaque Provider-authorized scalar expression over left/right.

    Expressions are trusted build inputs, like existing atomic recipes; runtime
    data can never inject source. Width/promotion semantics belong to Provider.
    """
    name: str
    code: int
    expression: str

    def __post_init__(self):
        identifier(self.name)
        if not 0 <= self.code <= 255 or not self.expression:
            raise ValueError('invalid primitive contract')


@dataclass(frozen=True)
class ScalarLoop:
    name: str
    scalar: ScalarType
    primitives: tuple[Primitive, ...]

    def __post_init__(self):
        identifier(self.name)
        if not self.primitives or len({p.code for p in self.primitives}) != len(self.primitives):
            raise ValueError('empty or duplicate primitive codes')
