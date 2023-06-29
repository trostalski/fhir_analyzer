from typing import Any, Callable, Union

from fhirpathpy import compile
import pandas as pd

from fhir_analyzer.fhirstore import Fhirstore


def evaluate_cond_fns(resource: dict, fns: list[dict[Callable, Callable]]) -> str:
    for fns_dic in fns:
        for cond_fn, targ_fn in fns_dic.items():
            if cond_fn(resource):
                res = targ_fn(resource)
                if res:
                    return res
    return None


class FeatureSelector:
    def __init__(self, fhirstore: Fhirstore):
        self._feature_names: list[str] = []
        self._feature_types: dict[str, str] = {}
        self._patient_features: list[dict[str, list[str]]] = {}
        self._fhirstore = fhirstore

    @property
    def feature_df(self):
        return pd.DataFrame(self._patient_features).T

    def _add_single_feature(
        self,
        feature_name: str,
        feature_type: str,
        target_resource_types: list[str],
        target_paths: dict[str, list[str]],
        conditional_target_paths: dict[str, list[dict[str, str]]] = None,
    ):
        if feature_name not in self._feature_names:
            self._add_feature_metadata(feature_name, feature_type)

        target_fns = {
            targ_n: [compile(targ_path) for targ_path in targ_paths]
            for targ_n, targ_paths in target_paths.items()
        }

        conditional_fns = (
            {
                targ_n: [
                    (compile(cond), compile(targ)) for cond, targ in cond_paths.items()
                ]
                for targ_n, cond_paths in conditional_target_paths.items()
            }
            if conditional_target_paths
            else {}
        )

        if not target_fns and not conditional_fns:
            raise ValueError("No target paths or conditional target paths provided.")

        self._process_target_resources(
            target_fns,
            conditional_fns,
            feature_name,
            target_resource_types,
        )

    def _add_feature_metadata(self, feature_name: str, feature_type: str):
        self._feature_names.append(feature_name)
        self._feature_types[feature_name] = feature_type

    def _process_target_resources(
        self,
        target_fns: list[Callable[[Any], Any]],
        conditional_fns: dict[
            str, list[tuple[Callable[[Any], bool], Callable[[Any], Any]]]
        ],
        feature_name: str,
        target_resource_types: list[str],
    ):
        for (
            patient_id,
            patient_resources,
        ) in self._fhirstore._patient_connections.items():
            self._patient_features.setdefault(patient_id, {})
            if feature_name not in self._patient_features[patient_id]:
                self._patient_features[patient_id][feature_name] = []
            for resource_type in target_resource_types:
                if resource_type in patient_resources:
                    for resource in patient_resources[resource_type]:
                        target = self._get_target(resource, target_fns, conditional_fns)
                        self._update_patient_features(patient_id, feature_name, target)

    def _get_target(
        self,
        resource: Any,
        target_fns: Union[list[Callable[[Any], Any]], dict[str, Callable[[Any], Any]]],
        conditional_fns: Union[
            list[tuple[Callable[[Any], bool], Callable[[Any], Any]]],
            dict[str, tuple[Callable[[Any], bool], Callable[[Any], Any]]],
        ],
    ):
        target = {}
        if conditional_fns:
            for targ_n, cond_fns in conditional_fns.items():
                temp_target = self._evaluate_conditional_functions(resource, cond_fns)
                if temp_target:
                    target[targ_n] = temp_target
        if not target:
            for targ_n, targ_fns in target_fns.items():
                temp_target = self._evaluate_target_functions(resource, targ_fns)
                if temp_target:
                    target[targ_n] = temp_target
                else:
                    target[targ_n] = None
        return target

    def _evaluate_conditional_functions(
        self,
        resource: Any,
        conditional_fns: list[tuple[Callable[[Any], bool], Callable[[Any], Any]]],
    ):
        for cond_fn, targ_fn in conditional_fns:
            if cond_fn(resource):
                return targ_fn(resource)[0]
        return None

    def _evaluate_target_functions(
        self, resource: Any, target_fns: list[Callable[[Any], Any]]
    ):
        for target_fn in target_fns:
            if not len(target_fn(resource)) > 0:
                continue
            target = target_fn(resource)[0]
            if target:
                return target
        return None

    def _update_patient_features(self, patient_id: str, feature_name: str, target: Any):
        if target:
            self._patient_features[patient_id][feature_name].append(target)
        else:
            self._patient_features[patient_id][feature_name].append(None)
