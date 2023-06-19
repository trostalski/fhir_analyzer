import statistics
import networkx as nx

from nxontology import NXOntology
from scipy.stats import norm

from fhir_analyzer.patient_similarity.internal_types import (
    CategoricalString,
    Numerical,
    CodedConcept,
    CodedNumerical,
)


def load_cc_graph(
    name: str,
    dir: str = "fhir_analyzer/patient_similarity/cc_graphs",
):
    G_default = nx.read_gpickle(f"{dir}/{name}.gpickle")
    G = NXOntology(G_default)
    G.freeze()
    print(f"Loaded {name} graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G


class Comparator:
    def __init__(self):
        pass

    def compare_categorical(
        self, feature1: list[CategoricalString], feature2: list[CategoricalString]
    ):
        """Computes the Jaccard similarity between two categorical features."""
        set1 = set([i.value for i in feature1])
        set2 = set([i.value for i in feature2])
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def compare_numerical(self, feature1: Numerical, feature2: Numerical):
        """Compute the mean euclidean distance between two numerical features."""
        nom = abs(feature1.value - feature2.value) - feature1.min_value
        denom = feature1.max_value - feature1.min_value
        return 1 - (nom / denom)

    def compare_coded_concepts(
        self,
        feature1: list[CodedConcept],
        feature2: list[CodedConcept],
        system: str,
        ic_metric: str = "intrinsic_ic_sanchez",
        cs_metric: str = "lin",
    ):
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
                similarity = self.graphs[system].similarity(code_a, code_b, ic_metric)
                similarity = getattr(similarity, cs_metric)
                node_sim_ab.append(similarity)
            node_sim.append(max(node_sim_ab, default=0))
        return node_sim

    def compare_coded_numerical_pair(
        self,
        feature1: CodedNumerical,
        feature2: CodedNumerical,
    ):
        if (
            any(val is None for val in (feature1.code_mean, feature2.code_std_dev))
            or feature1.code != feature2.code
        ):
            return None

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
        similarities = [
            self.compare_coded_numerical_pair(feature1, feature2)
            for a in feature1
            for b in feature2
            if a.code == b.code
        ]

        return statistics.mean(similarities) if similarities else None
