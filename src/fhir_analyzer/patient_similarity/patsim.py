from fhir_analyzer.feature_selector import FeatureSelector

from fhir_analyzer.fhirstore import Fhirstore
from fhir_analyzer.constants import (
    default_target_paths,
    default_system_paths,
    default_code_paths,
)
from fhir_analyzer.patient_similarity.comparator import Comparator
from fhir_analyzer.patient_similarity.internal_types import (
    CATEGORICAL_STRING,
    CODED_CONCEPT,
    CODED_NUMERICAL,
    NUMERICAL,
)


def get_default_target_paths(resource_types: list[str]) -> list[str]:
    result = []
    for resource_type in resource_types:
        if resource_type in default_target_paths:
            result.extend(default_target_paths[resource_type])
    return result


def get_default_code_paths(resource_types: list[str]) -> list[str]:
    result = []
    for resource_type in resource_types:
        if resource_type in default_code_paths:
            result.extend(default_code_paths[resource_type])
    return result


def get_default_system_paths(resource_types: list[str]) -> list[str]:
    result = []
    for resource_type in resource_types:
        if resource_type in default_system_paths:
            result.extend(default_system_paths[resource_type])
    return result


def validate_paths(paths: list[str], resource_types: list[str], default_fn, path_name):
    paths = default_fn(resource_types) if not paths else paths
    if not paths:
        raise ValueError(
            f"No {path_name} paths provided and no default paths for resource types."
        )


class Patsim:
    def __init__(self, fhirstore: Fhirstore = None):
        self._fhirstore = fhirstore if fhirstore else Fhirstore()
        self._feature_selector = FeatureSelector(self._fhirstore)

    def add_categorical_feature(
        self,
        name: str,
        resource_types: list[str] | str,
        target_paths: list[str] | str | None = None,
        conditional_target_paths: list[dict[str, str]] | dict[str, str] | None = None,
    ):
        if isinstance(resource_types, str):
            resource_types = [resource_types]
        if isinstance(target_paths, str):
            target_paths = [target_paths]
        validate_paths(
            paths=target_paths,
            default_fn=get_default_target_paths,
            resource_types=resource_types,
            path_name="target",
        )
        target_paths = {
            "value": target_paths,
        }
        if conditional_target_paths:
            conditional_target_paths = {
                "value": conditional_target_paths,
            }
        self._feature_selector._add_single_feature(
            feature_name=name,
            feature_type=CATEGORICAL_STRING,
            target_resource_types=resource_types,
            target_paths=target_paths,
            conditional_target_paths=conditional_target_paths,
        )

    def add_numerical_feature(
        self,
        name: str,
        resource_types: list[str] | str,
        target_paths: list[str] | str | None = None,
        conditional_target_paths: list[dict[str, str]] | dict[str, str] | None = None,
    ):
        if isinstance(resource_types, str):
            resource_types = [resource_types]
        if isinstance(target_paths, str):
            target_paths = [target_paths]
        validate_paths(
            paths=target_paths,
            default_fn=get_default_target_paths,
            resource_types=resource_types,
            path_name="value",
        )
        target_paths = {
            "value": target_paths,
        }
        if conditional_target_paths:
            conditional_target_paths["value"] = conditional_target_paths
        self._feature_selector._add_single_feature(
            feature_name=name,
            feature_type=NUMERICAL,
            target_resource_types=resource_types,
            target_paths=target_paths,
            conditional_target_paths=conditional_target_paths,
        )

    def add_coded_concept_feature(
        self,
        name: str,
        resource_types: list[str] | str,
        code_paths: list[str] | str | None = None,
        system_paths: list[str] | str | None = None,
        conditional_code_paths: list[dict[str, str]] | dict[str, str] | None = None,
        conditional_system_paths: list[dict[str, str]] | dict[str, str] | None = None,
    ):
        if isinstance(resource_types, str):
            resource_types = [resource_types]
        if isinstance(code_paths, str):
            code_paths = [code_paths]
        if isinstance(system_paths, str):
            system_paths = [system_paths]
        code_paths = (
            get_default_code_paths(resource_types) if not code_paths else code_paths
        )
        validate_paths(
            paths=code_paths,
            default_fn=get_default_code_paths,
            resource_types=resource_types,
            path_name="code",
        )
        system_paths = (
            (get_default_system_paths(resource_types))
            if not system_paths
            else system_paths
        )
        validate_paths(
            paths=system_paths,
            default_fn=get_default_system_paths,
            resource_types=resource_types,
            path_name="system",
        )
        target_paths = {
            "code": code_paths,
            "system": system_paths,
        }
        conditional_target_paths = {}
        if conditional_code_paths:
            conditional_target_paths["code"] = conditional_code_paths
        if conditional_system_paths:
            conditional_target_paths["system"] = conditional_system_paths
        conditional_target_paths = (
            conditional_target_paths if conditional_target_paths else None
        )
        self._feature_selector._add_single_feature(
            feature_name=name,
            feature_type=CODED_CONCEPT,
            target_resource_types=resource_types,
            target_paths=target_paths,
            conditional_target_paths=conditional_target_paths,
        )

    def add_coded_numerical_feature(
        self,
        name: str,
        resource_types: list[str] | str,
        value_paths: list[str] | str | None = None,
        code_paths: list[str] | str | None = None,
        conditional_value_paths: list[dict[str, str]] | dict[str, str] | None = None,
        conditional_code_paths: list[dict[str, str]] | dict[str, str] | None = None,
    ):
        if isinstance(resource_types, str):
            resource_types = [resource_types]
        if isinstance(code_paths, str):
            code_paths = [code_paths]
        if isinstance(value_paths, str):
            value_paths = [value_paths]
        code_paths = (
            get_default_code_paths(resource_types) if not code_paths else code_paths
        )
        validate_paths(
            paths=code_paths,
            default_fn=get_default_code_paths,
            resource_types=resource_types,
            path_name="code",
        )
        value_paths = (
            (get_default_system_paths(resource_types))
            if not value_paths
            else value_paths
        )
        validate_paths(
            paths=value_paths,
            default_fn=get_default_system_paths,
            resource_types=resource_types,
            path_name="system",
        )
        target_paths = {
            "code": code_paths,
            "value": value_paths,
        }
        conditional_target_paths = {}
        if conditional_code_paths:
            conditional_target_paths["code"] = conditional_code_paths
        if conditional_value_paths:
            conditional_value_paths["value"] = conditional_value_paths
        conditional_target_paths = (
            conditional_target_paths if conditional_target_paths else None
        )
        self._feature_selector._add_single_feature(
            feature_name=name,
            feature_type=CODED_NUMERICAL,
            target_resource_types=resource_types,
            target_paths=target_paths,
            conditional_target_paths=conditional_target_paths,
        )

    @property
    def feature_df(self):
        return self._feature_selector.feature_df

    def compute_similarities(self, *args, **kwargs):
        self._comparator = Comparator(feature_selector=self._feature_selector)
        return self._comparator._compute_similarities()

    def add_resources(self, resource: list[dict]):
        self._fhirstore.add_resources(resource)

    def add_bundle(self, bundle: dict):
        self._fhirstore.add_bundle(bundle)
