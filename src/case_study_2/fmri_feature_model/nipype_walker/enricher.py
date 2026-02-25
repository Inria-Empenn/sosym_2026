from nipype_walker import main_walker

def add_metadata(json, partof, excludes, requires):
    json["partof"] = json["partof"] + partof
    json["excludes"] = json["excludes"] + excludes
    json["requires"] = json["requires"] + requires
    return json

if __name__ == '__main__':
    json = main_walker.walk()["SPM_Preprocessing"]
    json['ApplyDeformations'] = add_metadata(json['ApplyDeformations'], ['spatial_normalization'], [], [])
    json['ApplyVDM'] = add_metadata(json['ApplyVDM'], [], [], [])
    json['Coregister'] = add_metadata(json['Coregister'], ['coregistration'], [], [])
    json['CreateWarped'] = add_metadata(json['CreateWarped'], [], [], [])
    json['DARTEL'] = add_metadata(json['DARTEL'], [], [], [])
    json['DARTELNorm2MNI'] = add_metadata(json['DARTELNorm2MNI'], [], [], [])
    json['FieldMap'] = add_metadata(json['FieldMap'], ['distorsion_correction'], [], [])
    json['MultiChannelNewSegment'] = add_metadata(json['MultiChannelNewSegment'], [], [], [])
    json['NewSegment'] = add_metadata(json['NewSegment'], ['segmentation'], [], [])
    json['Normalize'] = add_metadata(json['Normalize'], ['spatial_normalization'], ['Normalize12'], [])
    json['Normalize12'] = add_metadata(json['Normalize12'], ['spatial_normalization'], ['Normalize'], [])
    json['Realign'] = add_metadata(json['Realign'], ['motion_correction_realignment'], [], [])
    json['RealignUnwarp'] = add_metadata(json['RealignUnwarp'], ['motion_correction_realignment', 'distorsion_correction'], [], [])
    json['Segment'] = add_metadata(json['Segment'], ['segmentation'], [], [])
    json['SliceTiming'] = add_metadata(json['SliceTiming'], ['slice_timing_correction'], [], [])
    json['Smooth'] = add_metadata(json['Smooth'], [], ['spatial_smoothing'], [])
    json['VBMSegment'] = add_metadata(json['VBMSegment'], ['segmentation'], [], [])
    print(json)
