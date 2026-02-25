import inspect

from nipype.interfaces.base.core import BaseInterface

from nipype_walker import input_walker


def walk(cls):
    """

    Inspect a subclass of BaseInterface

    :param cls:
    :return:
    """

    if not issubclass(cls, BaseInterface):
        return {}

    for members in inspect.getmembers(cls):
        if members[0] == '__dict__':
            json = {"desc": "", "name": "", "partof": [], "features": {}, "excludes": [], "requires": []}
            attributes = members[1]

            if not attributes['__module__'].endswith('model') and not attributes['__module__'].endswith('preprocess') :
                continue

            if "__doc__" in attributes and attributes['__doc__'] is not None:
                json["desc"] = attributes['__doc__'].partition('\n')[0]
            if "_jobname" in attributes:
                json["name"] = attributes['_jobname']
            json["features"]["input"] = input_walker.walk(attributes['input_spec'])
            if 'output_spec' in attributes:
                json["features"]["output"] = input_walker.walk(attributes['output_spec'])
            return json
    return {}
