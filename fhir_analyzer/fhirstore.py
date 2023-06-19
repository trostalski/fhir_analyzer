from fhir_analyzer.helper import gather_references_for_resource


class Fhirstore:
    def __init__(self, bundle: dict = None, resources: list[dict] = None):
        self._resources = []
        self._patient_ids = []
        self._patient_connections = {}
        if bundle:
            self.validate_bundle_input(bundle)
            self._resources += [entry["resource"] for entry in bundle["entry"]]
        if resources:
            self.validate_resources_input(resources)
            self._resources += resources
        if len(self._resources) > 0:
            self._update_patient_dicts(self._resources)

    def _update_patient_dicts(self, resources: list[dict]):
        """resources should not exists already"""
        for resource in resources:
            resource_id = resource["id"]
            resource_type = resource.get("resourceType", None)
            if resource_type == "Patient":
                if not resource_id in self._patient_ids:
                    self._patient_ids.append(resource_id)
                self._patient_connections[resource_id] = {resource_type: [resource]}

        for resource in resources:
            resource_type = resource.get("resourceType", None)
            references = gather_references_for_resource(resource)
            for reference in references:
                if reference.reference in self._patient_ids:
                    patient_connection = self._patient_connections.setdefault(
                        reference.reference, {}
                    )
                    if not resource_type in patient_connection:
                        patient_connection[resource_type] = []
                    patient_connection[resource_type].append(resource)

    def add_bundle(self, bundle: dict):
        self.validate_bundle_input(bundle)
        new_resources = [
            entry["resource"]
            for entry in bundle["entry"]
            if not self._resource_exists(entry["resource"])
        ]
        if len(new_resources) == 0:
            return
        self._resources += new_resources
        self._update_patient_dicts(new_resources)

    def add_resources(self, resources: list[dict]):
        resources = [
            resource for resource in resources if not self._resource_exists(resource)
        ]
        if len(resources) == 0:
            return
        self.validate_resources_input(resources)
        self._resources += resources
        self._update_patient_dicts(resources)

    def _resource_exists(self, resource: dict) -> bool:
        for existing_resource in self._resources:
            if (
                existing_resource["id"] == resource["id"]
                and existing_resource["resourceType"] == resource["resourceType"]
            ):
                return True
        return False

    def validate_bundle_input(self, bundle: dict):
        if not isinstance(bundle, dict):
            raise ValueError("Bundle input is not a dict.")
        if not bundle.get("entry", None):
            raise ValueError("Bundle input does not contain entry.")

    def validate_resources_input(self, resources: list[dict]):
        if not isinstance(resources, list):
            raise ValueError("Input is not a list.")
        if len(resources) == 0:
            raise ValueError("Input is empty.")
        if not all(isinstance(resource, dict) for resource in resources):
            raise ValueError("Not all resource are of type dict.")
