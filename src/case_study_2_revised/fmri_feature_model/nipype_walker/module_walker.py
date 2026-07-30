import inspect

from nipype_walker import component_walker
from nipype.interfaces.base.core import BaseInterface


def walk(module):
    """

    Inspect a module

    :return:
    """

    json = {}

    for cls in inspect.getmembers(module):
        if inspect.isclass(cls[1]) and issubclass(cls[1], BaseInterface):
            name = cls[1].__name__
            component = component_walker.walk(cls[1])
            if component:
                json.update({name: component})
    return json
