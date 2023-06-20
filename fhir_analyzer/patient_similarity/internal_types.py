from dataclasses import dataclass

CODED_CONCEPT = "coded_concept"
CATEGORICAL_STRING = "categorical_string"
NUMERICAL = "numerical"
CODED_NUMERICAL = "coded_numerical"


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
    max_value: float
    min_value: float


@dataclass
class CodedNumerical:
    value: float
    code: str
    is_abnormal: bool
    code_mean: float
    code_std_dev: float
