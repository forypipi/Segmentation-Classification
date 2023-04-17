import os
import pickle
import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import CAM
import torch
from torch.utils.data import Dataset, DataLoader, dataset
from torch import nn
import torch.nn.functional as F


def split_train_val_test(clinical: pd.DataFrame, test_radio=0.1, kFold=3):
    train_set, _ = train_test_split(clinical, test_size=test_radio, stratify=clinical['class'], random_state=42)
    train_set.insert(clinical.shape[1], 'training_label', np.nan)
    X, y = train_set['name'], train_set['class']
    skf = StratifiedKFold(n_splits=kFold, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(X, y)):
        train_set.iloc[val_idx, 2] = [fold] * len(val_idx)
    train_set = train_set.drop("class", axis=1)
    clinical = clinical.merge(train_set, how='left', left_on='name', right_on='name')
    clinical['training_label'].fillna(-1, inplace=True)
    clinical['training_label'] = clinical['training_label'].astype('int')
    clinical.to_csv("./cls/data/training label.csv")
    return clinical

def normalized(img, range1=(0, 1), device="cuda"):
    img = torch.as_tensor(img, dtype=torch.float32, device=device)      # int32 to float32
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())    # to 0--1
    img = img * (range1[1] - range1[0]) + range1[0]
    return img

class MyData(Dataset):
    def __init__(self, data_label: pd.DataFrame, label_list: list, device='CPU', modality="PETCT", root_dir=r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'):
        self.data = data_label[data_label['training_label'].isin(label_list)][:4]
        self.data_label = data_label
        self.root_dir = root_dir
        self.device = device
        self.modality = modality

    def __getitem__(self, item):
        # sex_dict = {'F': 0, 'M': 1}
        label_dict = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'MELANOMA': 2, 'LUNG_CANCER': 3}
        patient_id = self.data.iloc[item, 1]
        diagnose = self.data.iloc[item, 0]
        patient_dir = os.path.join(self.root_dir, diagnose, patient_id)
        img_method = "tensor"
        if img_method == "sitk":
            ct_path = os.path.join(patient_dir, 'CT_nii', '1.nii')
            pet_path = os.path.join(patient_dir, 'PET_nii', '1.nii')
            ct_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), device=self.device)
            pet_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(pet_path)), device=self.device)
        elif img_method == 'tensor':
            ct_path = os.path.join(patient_dir, 'CT_pt', '1.pt')
            pet_path = os.path.join(patient_dir, 'PET_pt', '1.pt')
            ct_image = normalized(torch.load(ct_path), device=self.device)
            pet_image = normalized(torch.load(pet_path), device=self.device)
        elif img_method == "pickle":
            ct_path = os.path.join(patient_dir, 'ct_pickle', '1.pickle')
            pet_path = os.path.join(patient_dir, 'pet_pickle', '1.pickle')
            ct_f = open(ct_path, 'rb')
            pet_f = open(pet_path, 'rb')
            ct_image = normalized(pickle.load(ct_f), device=self.device)
            pet_image = normalized(pickle.load(pet_f), device=self.device)


        clinical = pd.read_csv(os.path.join(patient_dir, "clinical.csv"))

        label = torch.tensor(label_dict[clinical.loc[0, "label"]]).to(self.device)
        if self.modality == "PETCT":
            image = torch.stack((ct_image, pet_image))
        elif self.modality == "PET":
            image = pet_image.unsqueeze(dim=0)
        elif self.modality == "CT":
            image = ct_image.unsqueeze(dim=0)
        else:
            raise Exception(f"Wrong input: modality={self.modality}, must be one of 'PETCT'/'CT'/'PET'")
        return image, label, patient_dir
    
    def __len__(self):
        return len(self.data)


if __name__=="__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("running on GPU")
    else:
        device = torch.device('cpu')
        print("running on CPU")

    device = 'cpu'
    model = torch.load("/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/pipeline/cls/ResNet3D_PETCT.pt").to(device)
    model.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    model.fc = nn.Linear(64, 4)
    modality = "PETCT"
    root_dir = r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'

    kFold = 5
    labels = ["NEGATIVE", "LUNG_CANCER", "LYMPHOMA", "MELANOMA"]
    dfs = [pd.DataFrame({"class": [label] * len(os.listdir(os.path.join(root_dir, label))), "name": os.listdir(os.path.join(root_dir, label))}) for label in labels]
    Whole_df = pd.concat(dfs)

    # Get your input
    classes = split_train_val_test(clinical=Whole_df, kFold=kFold, test_radio=0.1)

    # Preprocess it for your chosen model
    validation = 0
    trainlist = list(range(validation))
    trainlist.extend(list(range(validation+1, kFold)))
    input_tensor = MyData(classes, trainlist, device, root_dir=root_dir, modality=modality)
    train_dataloader = DataLoader(dataset=input_tensor, batch_size=4, shuffle=True, drop_last=False, num_workers=0, persistent_workers=False)
    print(model)
    cam = CAM(model, input_shape=(2, 128, 128, 128))
    for img, _, path in train_dataloader:
        # Preprocess your data and feed it to the model
        out = model(img)
        print(out, out.argmax(dim=1).tolist())
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam(out.argmax(dim=1).tolist(), out)
        activation_map = F.interpolate(activation_map[0].unsqueeze(dim=1), scale_factor=(8, 8, 8), mode="trilinear")
        print(activation_map.shape)