import inspect
import itertools

import nipype.interfaces.spm.preprocess as spm_preprocess
from nipype.interfaces.base.core import BaseInterface
from nipype.interfaces.base.specs import TraitedSpec
from nipype.interfaces.base.traits_extension import InputMultiPath, File
from traits.trait_types import Range, Bool, Int, Float, String, Enum, List, Str

ignore_param = ['matlab_cmd', 'mfile', 'paths', 'trait_added', 'trait_modified', 'use_mcr', 'use_v8struct']


def walk(cls):
    """
    Inspect a subclass of BaseInterfaceInputSpec
    :param cls:
    :return:
    """
    json = {}
    if not issubclass(cls, TraitedSpec):
        return json
    for members in inspect.getmembers(cls):
        if members[0] == '__base_traits__':
            attributes = members[1]
            for key in attributes:
                if key in ignore_param:
                    continue
                name = key
                input_param = attributes[key].handler
                json[name] = walk_param(input_param)
    return json


def walk_param(trait):
    for member in inspect.getmembers(trait):
        if member[0] == '__dict__':
            attributes = member[1]
            json = parse_common_param_attributes(attributes)
            if isinstance(trait, Bool):
                json["type"] = "Boolean"
                json["default_value"] = parse_default_value(attributes)
            elif isinstance(trait, String) or isinstance(trait, Str):
                json["type"] = "String"
                json["default_value"] = parse_default_value(attributes)
            elif isinstance(trait, Int):
                json["type"] = "Integer"
                json["default_value"] = parse_default_value(attributes)
            elif isinstance(trait, Float):
                json["type"] = "Float"
                json["default_value"] = parse_default_value(attributes)
            elif isinstance(trait, InputMultiPath):
                json["type"] = "Files"
            elif isinstance(trait, File):
                json["type"] = "File"
            elif isinstance(trait, Enum):
                json.update(parse_enum(attributes))
                json["default_value"] = parse_default_value(attributes)
            elif isinstance(trait, Range):
                json.update(parse_range(attributes))
                json["default_value"] = parse_default_value(attributes)
            # elif isinstance(trait, List):
            #     print("List")
            return json
    return None


def parse_common_param_attributes(attributes):
    """
    Parse common values of a trait

    :param name:
    :param attributes:
    :return:
    """
    json = {"desc": "", "mandatory": False, "field": "", "default_value": None, "type": "Unsupported"}
    if "_metadata" in attributes:
        json["desc"] = attributes["_metadata"].get("desc", "")
        json["mandatory"] = attributes["_metadata"].get("mandatory", False)
        json["field"] = attributes["_metadata"].get("field", "")
        json["excludes"] = attributes["_metadata"].get("xor", [])
        json["requires"] = attributes["_metadata"].get("requires", [])
    return json

def parse_default_value(attributes):
    if "_metadata" in attributes:
        return attributes.get("default_value", None)


def parse_enum(attributes):
    oneof = [list([item for item in attributes.get("values", [])])]
    json = {
        "oneof": oneof
    }
    json["type"] = get_enum_type(json)
    return json


def get_enum_type(json):
    test = json["oneof"][0]
    if isinstance(test, float):
        return "Float"
    if isinstance(test, int):
        return "Integer"
    if isinstance(test, str):
        return "String"
    return "Unsupported"


def parse_range(attributes):
    """
    Parse a Range type trait
    :param attributes:
    :return:
    """
    json = {
        "value": attributes.get("_value", None),
        "low": attributes.get("_low", None),
        "high": attributes.get("_high", None),
    }
    json["type"] = get_range_type(json)
    return json


def get_range_type(json):
    """
    Infer range type (int, float) from value, low, high if set

    :param json:
    :return:
    """
    test = None
    if json["value"] is not None:
        test = json["value"]
    elif json["low"] is not None:
        test = json["low"]
    elif json["high"] is not None:
        test = json["high"]

    if isinstance(test, float):
        return "Float"
    if isinstance(test, int):
        return "Integer"
    return "Unsupported"
