from typing import Callable
from patient_similarity.feature_selector import FeatureSelector

from patient_similarity.fhirstore import Fhirstore
from patient_similarity.constants import (
    default_target_paths,
    default_system_paths,
    default_code_paths,
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
        validate_paths(
            paths=target_paths,
            default_fn=get_default_target_paths,
            resource_types=resource_types,
            path_name="target",
        )
        self._feature_selector._add_single_feature(
            name=name,
            target_resource_types=resource_types,
            target_paths=target_paths,
            target_name="value",
            conditional_target_paths=conditional_target_paths,
        )

    def add_numerical_feature(
        self,
        name: str,
        resource_types: list[str] | str,
        target_paths: list[str] | str | None = None,
        conditional_target_paths: list[dict[str, str]] | dict[str, str] | None = None,
    ):
        validate_paths(
            paths=target_paths,
            default_fn=get_default_target_paths,
            resource_types=resource_types,
            path_name="value",
        )
        self._feature_selector._add_single_feature(
            name=name,
            target_resource_types=resource_types,
            target_paths=target_paths,
            target_name="value",
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
        self._feature_selector._add_single_feature(
            name=name,
            target_resource_types=resource_types,
            target_paths=code_paths,
            target_name="code",
            conditional_target_paths=conditional_code_paths,
        )
        self._feature_selector._add_single_feature(
            name=name,
            target_resource_types=resource_types,
            target_paths=system_paths,
            target_name="system",
            conditional_target_paths=conditional_system_paths,
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
        self._feature_selector._add_single_feature(
            name=name,
            target_resource_types=resource_types,
            target_paths=value_paths,
            target_name="value",
            conditional_target_paths=conditional_value_paths,
        )
        self._feature_selector._add_single_feature(
            name=name,
            target_resource_types=resource_types,
            target_paths=code_paths,
            target_name="code",
            conditional_target_paths=conditional_code_paths,
        )

    def compute_similarities():
        pass

    def add_resources(self, resource: list[dict]):
        self._fhirstore.add_resources(resource)

    def add_bundle(self, bundle: dict):
        self._fhirstore.add_bundle(bundle)
