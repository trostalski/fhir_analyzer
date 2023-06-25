import pickle
import statistics
import re
import pkg_resources


from nxontology import NXOntology
import networkx as nx
import pandas as pd
from scipy.stats import norm

from fhir_analyzer.feature_selector import FeatureSelector

from fhir_analyzer.patient_similarity.internal_types import (
    CATEGORICAL_STRING,
    CODED_CONCEPT,
    CODED_NUMERICAL,
    NUMERICAL,
    CategoricalString,
    Numerical,
    CodedConcept,
    CodedNumerical,
)

SNOMED = "snomed"
ICD10 = "ICD-10"
LOINC = "LOINC"
RXNORM = "RxNorm"
UCUM = "UCUM"
ICD9 = "ICD-9"

SNOMED_GRAPH_NAME = "snomed_cc_graph.adjlist"
ICD10_GRAPH_NAME = "icd10_cc_graph.gpickle"


def load_nx_graph(
    name: str,
):
    G = pickle.load(
        pkg_resources.resource_stream(
            "fhir_analyzer.patient_similarity", f"nx_graphs/{name}.gpickle"
        )
    )
    G = NXOntology(G)
    G.freeze()
    print(
        f"Loaded {name} graph with {len(G.graph.nodes)} nodes and {len(G.graph.edges)} edges."
    )
    return G


class Comparator:
    def __init__(self, feature_selector: FeatureSelector = None):
        self._feature_selector = feature_selector
        self._numerical_stats = {}
        self._coded_numerical_stats = {}
        self._feature_dict = {}
        self._nx_graphs = {}
        self._add_type_data()
        self._build_feature_dict()
        self._sim_fns = {
            CATEGORICAL_STRING: self.compare_categorical,
            NUMERICAL: self.compare_numerical,
            CODED_CONCEPT: self.compare_coded_concepts,
            CODED_NUMERICAL: self.compare_coded_numerical,
        }

    def compare_categorical(
        self, feature1: list[CategoricalString], feature2: list[CategoricalString]
    ):
        """Computes the Jaccard similarity between two categorical features."""
        if len(feature1) == 0 or len(feature2) == 0:
            return None
        set1 = set([i.value for i in feature1])
        set2 = set([i.value for i in feature2])
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def compare_numerical(self, feature1: list[Numerical], feature2: list[Numerical]):
        """Compute the mean euclidean distance between two numerical features."""
        if len(feature1) != 1 or len(feature2) != 1:
            return None
        feature1 = feature1[0]
        feature2 = feature2[0]
        nom = abs(feature1.value - feature2.value) - feature1.min_value
        denom = feature1.max_value - feature1.min_value
        return 1 - (nom / denom)

    def compare_coded_concepts(
        self,
        feature1: list[CodedConcept],
        feature2: list[CodedConcept],
        ic_metric: str = "intrinsic_ic_sanchez",
        cs_metric: str = "lin",
    ):
        if len(feature1) == 0 or len(feature2) == 0:
            return None
        system = feature1[0].system
        system = self._resolve_system(system=system)

        node_sim_ab = self.calculate_node_similarities(
            feature1, feature2, system, ic_metric, cs_metric
        )
        node_sim_ba = self.calculate_node_similarities(
            feature2, feature1, system, ic_metric, cs_metric
        )

        if not node_sim_ab or not node_sim_ba:
            return 0

        factor = 1 / (len(node_sim_ab) + len(node_sim_ba))
        result = factor * (max(node_sim_ab, default=0) + max(node_sim_ba, default=0))
        return result

    def calculate_node_similarities(
        self,
        feature1: list[CodedConcept],
        feature2: list[CodedConcept],
        system: str,
        ic_metric: str,
        cs_metric: str,
    ) -> list[float]:
        node_sim = []
        for f1 in feature1:
            code_a = f1.code
            node_sim_ab = []
            for f2 in feature2:
                code_b = f2.code
                similarity = None
                try:
                    similarity = self._nx_graphs[system].similarity(
                        code_a, code_b, ic_metric
                    )
                except nx.NodeNotFound:
                    continue
                similarity = getattr(similarity, cs_metric)
                node_sim_ab.append(similarity)
            node_sim.append(max(node_sim_ab, default=0))
        return node_sim

    def compare_coded_numerical_pair(
        self,
        feature1: CodedNumerical,
        feature2: CodedNumerical,
    ):
        if feature1.is_abnormal or feature2.is_abnormal:
            mean = float(feature1.code_mean)
            std = float(feature1.code_std_dev)

            if std == 0:
                return None

            value1 = float(feature1.value)
            value2 = float(feature2.value)

            p1 = norm.cdf((value1 - mean) / std)
            p2 = norm.cdf((value2 - mean) / std)

            similarity = 1 - abs(p1 - p2)

            mean_percentile = (p1 + p2) / 2
            similarity *= 2 * abs(mean_percentile - 0.5)
            return similarity
        else:
            return None

    def compare_coded_numerical(
        self, feature1: list[CodedNumerical], feature2: list[CodedNumerical]
    ):
        if len(feature1) == 0 or len(feature2) == 0:
            return None
        similarities = [
            self.compare_coded_numerical_pair(a, b)
            for a in feature1
            for b in feature2
            if a.code == b.code
        ]

        return statistics.mean(similarities) if similarities else None

    def _add_type_data(self):
        for name, type in self._feature_selector._feature_types.items():
            if type == NUMERICAL:
                self._add_numerical_type_data(name)
            elif type == CODED_NUMERICAL:
                self._add_coded_numerical_type_data(name)

    def _add_numerical_type_data(self, name):
        values = []
        numerical_stats = {name: {"min_value": None, "max_value": None}}
        for _, features_dic in self._feature_selector._patient_features.items():
            for n, features in features_dic.items():
                if name == n:
                    for feature in features:
                        value = feature["value"]
                        if value is not None:
                            values.append(feature["value"])
        if values:
            min_value = min(values)
            max_value = max(values)
            if name not in numerical_stats:
                numerical_stats[name] = {
                    "min_value": min_value,
                    "max_value": max_value,
                }
        self._numerical_stats.update(numerical_stats)

    def _add_coded_numerical_type_data(self, name):
        code_values = {}
        code_stats = {name: {}}
        for _, features_dic in self._feature_selector._patient_features.items():
            for n, features in features_dic.items():
                if n == name:
                    for feature in features:
                        code = feature["code"]
                        value = feature["value"]
                        if code not in code_values:
                            code_values[code] = []
                        if value:
                            code_values[code].append(value)
        for code, values in code_values.items():
            if code and values:
                code_stats[name][code] = {
                    "mean": statistics.mean(values),
                    "std_dev": statistics.stdev(values),
                }
            else:
                code_stats[name][code] = {"mean": None, "std_dev": None}
        self._coded_numerical_stats.update(code_stats)

    def _build_feature_dict(self):
        updated_feature_dic = {}
        for (
            patient_id,
            features_dic,
        ) in self._feature_selector._patient_features.items():
            updated_feature_dic[patient_id] = {}
            for name, features in features_dic.items():
                if self._feature_selector._feature_types[name] == NUMERICAL:
                    parsed_features = [
                        Numerical(
                            value=feature["value"],
                            min_value=self._numerical_stats[name]["min_value"],
                            max_value=self._numerical_stats[name]["max_value"],
                            feature_name=name,
                        )
                        for feature in features
                        if feature["value"] is not None
                        and self._numerical_stats[name]["min_value"] is not None
                        and self._numerical_stats[name]["max_value"] is not None
                    ]
                elif self._feature_selector._feature_types[name] == CODED_NUMERICAL:
                    parsed_features = [
                        CodedNumerical(
                            code=feature["code"],
                            value=feature["value"],
                            code_mean=self._coded_numerical_stats[name][
                                feature["code"]
                            ]["mean"],
                            code_std_dev=self._coded_numerical_stats[name][
                                feature["code"]
                            ]["std_dev"],
                            is_abnormal=feature["is_abnormal"]
                            if "is_abnormal" in feature
                            else True,
                            feature_name=name,
                        )
                        for feature in features
                        if feature["value"] is not None
                        and feature["code"] is not None
                        and self._coded_numerical_stats[name][feature["code"]]["mean"]
                        is not None
                        and self._coded_numerical_stats[name][feature["code"]][
                            "std_dev"
                        ]
                        is not None
                    ]
                elif self._feature_selector._feature_types[name] == CODED_CONCEPT:
                    parsed_features = [
                        CodedConcept(
                            code=feature["code"],
                            system=feature["system"],
                            feature_name=name,
                        )
                        for feature in features
                        if feature["code"] is not None and feature["system"] is not None
                    ]
                elif self._feature_selector._feature_types[name] == CATEGORICAL_STRING:
                    parsed_features = [
                        CategoricalString(
                            value=feature["value"],
                            feature_name=name,
                        )
                        for feature in features
                        if feature["value"] is not None
                    ]
                updated_feature_dic[patient_id].update({name: parsed_features})
        self._feature_dict.update(updated_feature_dic)

    def _compute_similarities(self):
        sim_df_data = {}
        result_dict = {}
        for patient_id1, feature_dic1 in self._feature_dict.items():
            for feat_name, features in feature_dic1.items():
                feat_type = self._feature_selector._feature_types[feat_name]
                sim_fn = self._sim_fns[feat_type]
                sim_df_data.setdefault(feat_name, {})
                if patient_id1 not in sim_df_data[feat_name]:
                    sim_df_data[feat_name][patient_id1] = {}
                for patient_id2, feature_dic2 in self._feature_dict.items():
                    if patient_id2 not in sim_df_data[feat_name][patient_id1]:
                        sim_df_data[feat_name][patient_id1][patient_id2] = {}
                    if patient_id1 == patient_id2:
                        sim_df_data[feat_name][patient_id1][patient_id2] = 1
                    else:
                        sim_df_data[feat_name][patient_id1][patient_id2] = sim_fn(
                            features, feature_dic2[feat_name]
                        )
        for feat_name, data in sim_df_data.items():
            result_dict.update({feat_name: pd.DataFrame(data)})
        return result_dict

    def _resolve_system(self, system: str):
        system = re.sub(r"\W+", "", system)
        system = system.lower()
        resolved_system = None
        if "snomed" in system:
            resolved_system = SNOMED
        elif "icd10" in system:
            resolved_system = ICD10
        elif "loinc" in system:
            resolved_system = LOINC
        elif "rxnorm" in system:
            resolved_system = RXNORM
        elif "ucum" in system:
            resolved_system = UCUM
        elif "icd9" in system:
            resolved_system = ICD9
        else:
            raise ValueError(f"Unknown system: {resolved_system}")
        if resolved_system not in self._nx_graphs:
            self._nx_graphs[resolved_system] = load_nx_graph(name=resolved_system)
        return resolved_system
