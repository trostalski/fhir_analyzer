from dataclasses import dataclass

CODED_CONCEPT = "coded_concept"
CATEGORICAL_STRING = "categorical_string"
NUMERICAL = "numerical"
CODED_NUMERICAL = "coded_numerical"


class CodedConcept:
    code: str
    system: str

    def __init__(self, code: str, system: str):
        self.code = code
        self.system = system

    def __str__(self):
        return f"CodedConcept({self.code}, {self.system})"


class CategoricalString:
    value: str

    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return f"CategoricalString({self.value})"


class Numerical:
    value: float
    max_value: float
    min_value: float

    def __init__(self, value: float, max_value: float, min_value: float):
        self.value = value
        self.max_value = max_value
        self.min_value = min_value

    def __str__(self):
        return f"Numerical({self.value}, {self.max_value}, {self.min_value})"


class CodedNumerical:
    value: float
    code: str
    is_abnormal: bool
    code_mean: float
    code_std_dev: float

    def __init__(
        self,
        value: float,
        code: str,
        code_mean: float,
        code_std_dev: float,
        is_abnormal: bool,
    ):
        is_abnormal = is_abnormal if is_abnormal is not None else True
        self.value = value
        self.code = code
        self.code_mean = code_mean
        self.code_std_dev = code_std_dev
        self.is_abnormal = is_abnormal

    def __str__(self):
        return f"CodedNumerical({self.value}, {self.code}, {self.code_mean}, {self.code_std_dev}, {self.is_abnormal})"
