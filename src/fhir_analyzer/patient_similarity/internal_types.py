from dataclasses import dataclass

CODED_CONCEPT = "coded_concept"
CATEGORICAL_STRING = "categorical_string"
NUMERICAL = "numerical"
CODED_NUMERICAL = "coded_numerical"


class CodedConcept:
    def __init__(self, code: str, system: str, feature_name: str):
        if not isinstance(code, str):
            raise ValueError(
                f"Feature code for {feature_name} expected str, got {type(code)}: {code}"
            )
        self.code = code
        self.system = system

    def __str__(self):
        return f"CodedConcept({self.code}, {self.system})"


class CategoricalString:
    def __init__(self, value: str, feature_name: str):
        if not isinstance(value, str):
            raise ValueError(
                f"Feature value for {feature_name} expected str, got {type(value)}: {value}"
            )
        self.value = value

    def __str__(self):
        return f"CategoricalString({self.value})"


class Numerical:
    def __init__(
        self, value: float, max_value: float, min_value: float, feature_name: str
    ):
        if not isinstance(value, float):
            raise ValueError(
                f"Feature value for {feature_name} expected float, got {type(value)}: {value}"
            )
        self.value = value
        self.max_value = max_value
        self.min_value = min_value

    def __str__(self):
        return f"Numerical({self.value}, {self.max_value}, {self.min_value})"


class CodedNumerical:
    def __init__(
        self,
        value: float,
        code: str,
        code_mean: float,
        code_std_dev: float,
        feature_name: str,
        is_abnormal: bool | None = None,
    ):
        if not isinstance(value, float):
            raise ValueError(
                f"Feature value for {feature_name} expected float, got {type(value)}: {value}"
            )
        if not isinstance(code, str):
            raise ValueError(
                f"Feature code for {feature_name} expected str, got {type(code)}: {code}"
            )
        is_abnormal = is_abnormal if is_abnormal is not None else True
        self.value = value
        self.code = code
        self.code_mean = code_mean
        self.code_std_dev = code_std_dev
        self.is_abnormal = is_abnormal

    def __str__(self):
        return f"CodedNumerical({self.value}, {self.code}, {self.code_mean}, {self.code_std_dev}, {self.is_abnormal})"
