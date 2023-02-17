import collections
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


def normalized(img: sitk.Image, range1=(0, 1)):
    img = sitk.GetArrayFromImage(img)
    img = img.astype(np.float32)      # int32 to float32
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())    # to 0--1
    img = img * (range1[1] - range1[0]) + range1[0]
    return sitk.GetImageFromArray(img.astype(np.uint8))

def ReadDcms(study_no: int, root_dir=r'E:\dataset\preprocessed'):
    sex_dict = {'F': 0, 'M': 1}
    patient_dir = os.path.join(root_dir, str(study_no))
    ct_path = os.path.join(patient_dir, 'CT_nii', '1.nii')
    pet_path = os.path.join(patient_dir, 'PET_nii', '1.nii')
    seg_path = os.path.join(patient_dir, 'seg_nii', '1.nii')
    ct_image = normalized(sitk.ReadImage(ct_path), (0, 255))
    pet_image = normalized(sitk.ReadImage(pet_path), (0, 255))
    seg_image = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    seg_image[seg_image == 255] = 1
    seg_image = sitk.GetImageFromArray(seg_image)
    clinical = pd.read_csv(os.path.join(patient_dir, "clinical.csv"))
    age = int(clinical.loc[0, "age"])
    sex = clinical.loc[0, "sex"]
    if isinstance(sex, str):
        sex = sex_dict[sex]
    else:
        sex = 2
    label = clinical.loc[0, "label"]
    return pet_image, ct_image, seg_image, age, sex, label

classes = pd.read_csv(r'E:\dataset\target.csv', index_col=0)
clinical = pd.read_csv(r'E:\dataset\Clinical Metadata FDG PET_CT Lesions.csv')
settings = dict()
settings['binWidth'] = 25
features = pd.read_csv("../data/features.csv", index_col=0)
for i in range(len(os.listdir('E:\dataset\preprocessed'))):
    if i < 847:
        continue
    print("-"*10, f"start pyradiomics for patient {i}", "-"*10)
    pet, ct, seg, age, sex, label = ReadDcms(i)
    print(label, sitk.GetArrayFromImage(seg).max())
    if sitk.GetArrayFromImage(seg).max() == 1:
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.enableAllFeatures()
        pet_result = extractor.computeFeatures(pet, seg, imageTypeName="original")
        pet_result = collections.OrderedDict([("pet_" + k, v) for k, v in pet_result.items()])
        ct_result = extractor.computeFeatures(ct, seg, imageTypeName="original")
        ct_result = collections.OrderedDict([("ct_" + k, v) for k, v in ct_result.items()])
        result = dict()
        result.update(pet_result)
        result.update(ct_result)
        result['label'] = label
        features = features.append(pd.DataFrame(pd.DataFrame(result, index=[i])))
        # for key, value in result.items():
        #     print('\t', key, ':', value)
        print(features.tail(10))
        features.to_csv("../data/features.csv")
