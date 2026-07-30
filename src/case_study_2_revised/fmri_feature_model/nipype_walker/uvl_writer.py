import json

from nipype_walker import main_walker

indent = "    "
newline = "\n"
output = "../model/uvl/"

def write():
    raw = main_walker.walk()
    json_object = json.load(raw)


def format_desc(desc):
    if not desc:
        return ""
    return " // " + " ".join(desc.replace("\n", " ").split())


def get_uvl_type(ttype):
    if ttype == "Float":
        return "Real"
    return ttype

def write_feature(isMandatory, level, features):
    uvl = ""
    if isMandatory:
        uvl += indent * level + "mandatory" + newline
    else:
        uvl += indent * level + "optional" + newline
    level += 1
    for key in features:
        param = features[key]
        uvl += indent * level + " " + key + format_desc(param["desc"]) + newline
        if "oneof" in param:
            level += 1
            uvl += indent * level + "Alternative" + newline
            level += 1
            for value in param["oneof"]:
                uvl += indent * level + str(value) + newline
            level -= 2
    level -= 1
    return uvl

def write_features(name, json):
    uvl = "features" + newline
    level = 1
    uvl += indent * level + "\"" + name + "\" {abstract}" + newline
    level += 1
    uvl += indent * level + "optional" + newline
    level += 1
    for key in json:
        if not json[key]:
            continue
        desc = json[key]['desc']
        params = json[key]['features']['input']

        uvl += indent * level + key + format_desc(desc) + newline  # step
        level += 1

        optional = {}
        mandatory = {}
        for key in params:
            param = params[key]
            if param['mandatory']:
                mandatory.update({key: param})
            else:
                optional.update({key: param})

        if mandatory:
            uvl += write_feature(True, level, mandatory)

        if optional:
            uvl += write_feature(False, level, optional)

        level -= 1
    level -= 1
    return uvl

def write_constraints(json):
    uvl = "constraints" + newline
    level = 1
    for key in json:
        if not json[key]:
            continue
        params = json[key]["features"]
        for key in params:
            param = params[key]

            hasLow = "low" in param and param['low'] is not None
            hasHigh = "high" in param and param['high'] is not None
            hasExcludes = "excludes" in param and param['excludes']
            hasRequires = "requires" in param and param['requires']

            if hasLow or hasHigh:
                uvl += indent * level

                if hasLow:
                    uvl += key + " >= " + str(param['low'])

                if hasLow and hasHigh:
                    uvl += " & "
                else:
                    uvl += newline

                if hasHigh:
                    uvl += key + " <= " + str(param['high']) + newline

            if hasExcludes:
                for feat in param['excludes']:
                    uvl += indent * level + key + " => !" + feat + newline

            if hasRequires:
                for feat in param['requires']:
                    uvl += indent * level + key + " => " + feat + newline

    return uvl


if __name__ == '__main__':
    json = main_walker.walk()
    for key in json:
        steps = json[key]
        uvl = write_features(key, steps) + write_constraints(steps)
        f = open(output + key + '.uvl', "w")
        f.write(uvl)
        f.close()
