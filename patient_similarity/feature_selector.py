from typing import Callable

from patient_similarity.fhirstore import Fhirstore
from fhirpathpy import compile


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
        self._feature_names = []
        self._patient_features = {}
        self._fhirstore = fhirstore

    def _add_single_feature(
        self,
        name: str,
        target_resource_types: list[str],
        target_paths: list[str],
        target_name: str,
        conditional_target_paths: list[dict[str, str]] = None,
    ):
        if name not in self._feature_names:
            self._feature_names.append(name)
        target_fns = [compile(target_path) for target_path in target_paths]
        conditional_fns = []
        if conditional_target_paths:
            for cond_p, targ_p in conditional_target_paths.items():
                conditional_fns.append((compile(cond_p), compile(targ_p)))

        if not target_fns and not conditional_fns:
            raise ValueError("No target paths or conditional target paths provided.")

        for (
            patient_id,
            patient_resources,
        ) in self._fhirstore._patient_connections.items():
            self._patient_features.setdefault(patient_id, {})
            for resource_type in target_resource_types:
                if resource_type not in patient_resources:
                    continue
                for resource in patient_resources[resource_type]:
                    target = None
                    if len(conditional_fns) > 0:
                        target = evaluate_cond_fns(resource, conditional_fns)
                    if not target:
                        for target_fn in target_fns:
                            target = target_fn(resource)
                            if target:
                                break
                    if target:
                        if not name in self._patient_features[patient_id]:
                            self._patient_features[patient_id][name] = {}
                        self._patient_features[patient_id][name][target_name] = target
