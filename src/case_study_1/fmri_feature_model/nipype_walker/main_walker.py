import json

import nipype.interfaces.afni
import nipype.interfaces.fsl
import nipype.interfaces.spm

from nipype_walker import module_walker

output = "../model/json/nipype.json"


def walk():
    """

    Inspect NiPype to extract features

    :return: dict
    """

    models = {"AFNI_Preprocessing": module_walker.walk(nipype.interfaces.afni.preprocess),
              "AFNI_Analysis": module_walker.walk(nipype.interfaces.afni.model),
              "FSL_Preprocessing": module_walker.walk(nipype.interfaces.fsl.preprocess),
              "FSL_Analysis": module_walker.walk(nipype.interfaces.fsl.model),
              "SPM_Preprocessing": module_walker.walk(nipype.interfaces.spm.preprocess),
              "SPM_Analysis": module_walker.walk(nipype.interfaces.spm.model)
              }

    return models


if __name__ == '__main__':
    parsed = json.loads(json.dumps(walk()))
    f = open(output, "w")
    f.write(json.dumps(parsed, indent=4))
    f.close()
    print(json.dumps(parsed, indent=4))
