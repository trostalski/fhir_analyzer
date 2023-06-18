from dataclasses import dataclass


@dataclass
class CodedConcept:
    code: str
    system: str


@dataclass
class CategoricalString:
    value: str


@dataclass
class Numerical:
    value: float


@dataclass
class CodedNumerical:
    value: float
    code: str
