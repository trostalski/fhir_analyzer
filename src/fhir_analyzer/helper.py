from typing import Generator, Iterable
from urllib.parse import urlparse
from uuid import UUID
from fhir.resources.reference import Reference

from fhir_analyzer.constants import RESOURCE_LIST


def get_references_generator(input: Iterable) -> Generator[Reference, None, None]:
    """Returns a generator for all values in a dictionary of the specified key.
    E.g. this is used to extract all references of a FHIR resource."""

    target_key = "reference"
    if isinstance(input, list):
        for item in input:
            yield from get_references_generator(item)

    if isinstance(input, dict):
        for k, v in input.items():
            if k == target_key:
                yield Reference(**input)
            else:
                yield from get_references_generator(v)


def gather_references_for_resource(resource: dict) -> list[Reference]:
    result = []
    for reference in get_references_generator(resource):
        new_reference = reference
        if is_absolute_or_relative_ref(reference.reference):
            new_reference.reference = reference.reference.split("/")[-1]
            new_reference.type = reference.reference.split("/")[-2]
        elif is_uuid(reference.reference):
            new_reference.reference = get_id_from_uuid(reference.reference)
        result.append(new_reference)
    return result


def is_absolute_or_relative_ref(id: str):
    """Checks if the provided id is a relative id (e.g. Patient/123)"""
    result = False
    id = urlparse(id).path  # extract path from url
    path_elements = id.split("/")
    if len(path_elements) >= 2:  # format [RESOURCE]/[ID]
        resource_type = path_elements[-2].capitalize()
        if resource_type in RESOURCE_LIST:
            result = True
    return result


def is_uuid(id: str):
    try:
        UUID(id)
        return True
    except ValueError:
        return False


def get_id_from_uuid(uuid: str) -> str:
    """Extracts the id from a uuid"""
    return uuid.split(":")[-1]


def inputs_to_lists(*args):
    """Converts all inputs to lists"""
    return tuple([arg if isinstance(arg, list) else [arg] for arg in args])
